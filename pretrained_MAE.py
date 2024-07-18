import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
from src.utils import set_seed

# データセットクラス
class PreprocessedEEGImageDataset(Dataset):
    def __init__(self, eeg_data_file, image_paths_file=None, labels_file=None):
        self.eeg_data = torch.load(eeg_data_file)
        
        if image_paths_file:
            with open(image_paths_file, 'r') as f:
                self.image_paths = f.read().splitlines()
            self.image_base_dir = r'C:\Users\controllab\dl_lecture_competition_pub\data\Images'
            self.valid_image_paths = [os.path.join(self.image_base_dir, path) for path in self.image_paths if os.path.isfile(os.path.join(self.image_base_dir, path))]
        else:
            self.image_paths = None

        if labels_file:
            self.labels = torch.load(labels_file)
        else:
            self.labels = None

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        if eeg.shape[1] == 281:
            eeg = eeg[:, :-1]  # 最後の要素を削除して280にする
        if self.image_paths:
            image_path = self.valid_image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            if self.labels is not None:
                return eeg, image, self.labels[idx]
            else:
                return eeg, image
        else:
            if self.labels is not None:
                return eeg, self.labels[idx]
            else:
                return eeg, None

# PatchEmbedding クラス
class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.Linear(patch_size * in_channels, embed_dim)
        )

    def forward(self, x):
        return self.to_patch_embedding(x)

# モデルクラス
class MAE(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, hidden_dim, dropout_rate):
        super(MAE, self).__init__()
        self.encoder = MAE_Encoder(seq_len=280, patch_size=10, emb_dim=embed_dim, num_layer=num_layers,
                                   heads=num_heads, dim_head=embed_dim // num_heads, mlp_dim=hidden_dim, mask_ratio=0.5, dropout=dropout_rate)
        self.decoder = MAE_Decoder(seq_len=280, patch_size=10, emb_dim=embed_dim, num_layer=4,
                                   heads=num_heads, dim_head=embed_dim // num_heads, mlp_dim=hidden_dim, dropout=dropout_rate)

    def forward(self, x, mask_ratio):
        features, backward_indexes = self.encoder(x)
        rec_img, mask = self.decoder(features, backward_indexes)
        return rec_img, mask

# Encoder クラス
class MAE_Encoder(nn.Module):
    def __init__(self, seq_len, patch_size, emb_dim, num_layer, heads, dim_head, mlp_dim, mask_ratio, dropout):
        super(MAE_Encoder, self).__init__()
        num_patches = seq_len // patch_size

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patchify = PatchEmbedding(seq_len, patch_size, 271, emb_dim)
        self.transformer = nn.Sequential(*[Block(emb_dim, heads, dim_head, mlp_dim, dropout) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = patches + self.pos_embedding[:, 1:, :]
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = torch.cat([self.cls_token.repeat(patches.shape[0], 1, 1), patches], dim=1)
        patches = patches + self.pos_embedding[:, :patches.shape[1], :]
        features = self.layer_norm(self.transformer(patches))
        return features, backward_indexes

# Decoder クラス
class MAE_Decoder(nn.Module):
    def __init__(self, seq_len, patch_size, emb_dim, num_layer, heads, dim_head, mlp_dim, dropout):
        super(MAE_Decoder, self).__init__()
        num_patches = seq_len // patch_size

        self.mask_token = torch.nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1 + 1, emb_dim))
        self.transformer = nn.Sequential(*[Block(emb_dim, heads, dim_head, mlp_dim, dropout) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.to_img = nn.Sequential(
            nn.Linear(emb_dim, patch_size * 271),
            Rearrange('b n (p c) -> b c (n p)', p=patch_size, c=271)
        )
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x, backward_indexes):
        B, N, D = x.shape
        mask_tokens = self.mask_token.repeat(B, backward_indexes.shape[1] - N, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = take_indexes(x_, backward_indexes)
        x_ = x_ + self.pos_embedding[:, :x_.shape[1], :]
        x_ = self.layer_norm(self.transformer(x_))
        x_ = self.to_img(x_)
        mask = torch.ones_like(x_)
        return x_, mask

# Attention, Block, FFN クラス
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ffn = FFN(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PatchShuffle(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches):
        B, N, dim = patches.shape
        remain_N = int(N * (1 - self.ratio))
        indexes = [random_indexes(N) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).T.to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).T.to(patches.device)
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:, :remain_N, :]
        return patches, forward_indexes, backward_indexes

def random_indexes(size):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, dim=1, index=indexes.unsqueeze(2).repeat(1, 1, sequences.shape[-1]))

# 損失関数
def mae_loss(recon_eeg, eeg, mask):
    loss = nn.MSELoss(reduction='none')(recon_eeg, eeg)
    loss = (loss * mask).sum() / mask.sum()
    return loss

def custom_collate(batch):
    eegs = [item[0] for item in batch]
    labels = [item[2] for item in batch] if len(batch[0]) == 3 else None
    eegs = torch.stack(eegs, dim=0)
    if labels is not None:
        labels = torch.tensor(labels)
        return eegs, labels
    return eegs

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="EEG-classification")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_prob = 0.5

    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": True, "collate_fn": custom_collate }
    train_image_paths_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\train_image_paths.txt'
    train_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_train_X.pt'
    train_labels_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_train_y.pt'
    val_image_paths_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\val_image_paths.txt'
    val_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_val_X.pt'
    val_labels_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_val_y.pt'
    test_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_test_X.pt'

    train_dataset = PreprocessedEEGImageDataset(train_eeg_data_file, train_image_paths_file, train_labels_file)
    val_dataset = PreprocessedEEGImageDataset(val_eeg_data_file, val_image_paths_file, val_labels_file)
    test_dataset = PreprocessedEEGImageDataset(test_eeg_data_file)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    sample_eeg, _ = next(iter(train_loader))
    seq_len = sample_eeg.shape[1]
    input_dim = sample_eeg.shape[0]
    
    # モデルインスタンス化時にパラメータを直接指定する
    eeg_encoder = MAE(input_dim=input_dim, embed_dim=256, num_layers=6, num_heads=8, hidden_dim=512, dropout_rate=0.2).to(device)
    eeg_encoder.seq_len = seq_len

    optimizer = torch.optim.AdamW([
        {'params': eeg_encoder.parameters(), 'lr': args.lr, 'weight_decay': 1e-2},
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, val_loss = 0.0, 0.0
        
        eeg_encoder.train()
        for eeg, labels in tqdm(train_loader, desc="Train"):
            eeg = eeg.to(device)
            
            recon_eeg, mask = eeg_encoder(eeg, mask_ratio=mask_prob)
            optimizer.zero_grad()
            loss = mae_loss(recon_eeg, eeg, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        eeg_encoder.eval()
        with torch.no_grad():
            for eeg, labels in tqdm(val_loader, desc="Validation"):
                eeg = eeg.to(device)
                
                recon_eeg, mask = eeg_encoder(eeg, mask_ratio=mask_prob)
                loss = mae_loss(recon_eeg, eeg, mask)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
        
        torch.save({
            'eeg_encoder': eeg_encoder.encoder.state_dict(),
            'eeg_decoder': eeg_encoder.decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(logdir, "model_last.pt"))

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_loss:
            cprint("New best.", "cyan")
            torch.save({
                'eeg_encoder': eeg_encoder.encoder.state_dict(),
                'eeg_decoder': eeg_encoder.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(logdir, "model_best.pt"))
            best_loss = val_loss

if __name__ == "__main__":
    run()
