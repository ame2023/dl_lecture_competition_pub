import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from termcolor import cprint
import hydra
from omegaconf import DictConfig
import wandb
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
import open_clip
from pretrained_MAE2 import MAE, PatchEmbedding  
from src.utils import set_seed

# CLIPモデルのロード
ViT_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')

def load_pretrained_mae(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.encoder.load_state_dict(checkpoint['eeg_encoder'])
    model.decoder.load_state_dict(checkpoint['eeg_decoder'])
    print("Pretrained MAE model loaded successfully.")

def save_fc_parameters(model, checkpoint_path):
    torch.save({
        'fc': model.fc.state_dict()
    }, checkpoint_path)
    print("FC layer parameters saved successfully.")

def load_fc_parameters(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.fc.load_state_dict(checkpoint['fc'])
    print("FC layer parameters loaded successfully.")

def cosine_similarity_loss(image_features, eeg_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    eeg_features = F.normalize(eeg_features, dim=-1)
    
    logits_per_image = torch.matmul(image_features, eeg_features.t()) / temperature
    logits_per_eeg = logits_per_image.t()
    
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size, device=image_features.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_e = F.cross_entropy(logits_per_eeg, labels)
    
    loss = (loss_i + loss_e) / 2.0
    return loss

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

# モデルクラス
class ModifiedMAE(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, hidden_dim, dropout_rate, output_dim):
        super(ModifiedMAE, self).__init__()
        self.mae = MAE(input_dim=input_dim, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),# 256->512
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        initialize_weights(self.fc)
        for param in self.fc.parameters():
            param.requires_grad = True  # fc層のみを最適化対象に設定
        for param in self.mae.encoder.parameters():
            param.requires_grad = False
        for param in self.mae.decoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        features, _ = self.mae.encoder(x)
        features = features[:, 0, :]  # cls_tokenの特徴量を使用
        x = self.fc(features)
        return x

def initialize_weights(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def custom_collate(batch):
    eegs = [item[0] for item in batch]
    images = [item[1] for item in batch] if batch[0][1] is not None else None
    labels = [item[2] for item in batch] if len(batch[0]) == 3 else None
    
    eegs = torch.stack(eegs, dim=0)
    if images is not None:
        images = [preprocess_train(image) for image in images]
        images = torch.stack(images, dim=0)
    if labels is not None:
        labels = torch.tensor(labels)
        return eegs, images, labels
    return eegs, images

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="EEG-classification")

    device = args.device

    # データロード
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": True, "collate_fn": custom_collate}
    train_image_paths_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\train_image_paths.txt'
    train_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_train_X.pt'
    train_labels_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_train_y.pt'
    val_image_paths_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\val_image_paths.txt'
    val_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_val_X.pt'
    val_labels_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_val_y.pt'

    train_dataset = PreprocessedEEGImageDataset(train_eeg_data_file, train_image_paths_file, train_labels_file)
    val_dataset = PreprocessedEEGImageDataset(val_eeg_data_file, val_image_paths_file, val_labels_file)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    # モデル定義
    eeg_encoder = ModifiedMAE(input_dim=271, embed_dim=256, num_layers=6, num_heads=8, hidden_dim=512, dropout_rate=0.2, output_dim=512).to(device)  # 出力次元を512に設定
    img_encoder = ViT_model.to(device)

    # 事前学習したMAEモデルのパラメータをロード
    pretrained_mae_path = r'C:\Users\controllab\dl_lecture_competition_pub\src\model_best_122epo.pt'
    load_pretrained_mae(eeg_encoder.mae, pretrained_mae_path)
    """"""
    """"""
    """
    """
    # 事前学習したfc層のパラメータをロード
    fc_checkpoint_path = os.path.join(logdir, r"C:\Users\controllab\dl_lecture_competition_pub\src\encoder_fc_best_7epo.pt")
    if os.path.exists(fc_checkpoint_path):
        load_fc_parameters(eeg_encoder, fc_checkpoint_path)

    # オプティマイザ定義
    optimizer = torch.optim.AdamW([
        {'params': eeg_encoder.fc.parameters(), 'lr': args.lr, 'weight_decay': 1e-2},  # fc層のみを最適化対象に追加
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    
    best_cosine_sim = float('-inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, val_loss = 0.0, 0.0
        train_cosine_sim, val_cosine_sim = 0.0, 0.0
        
        eeg_encoder.train()
        img_encoder.eval()
        for eeg, images, labels in tqdm(train_loader, desc="Train"):
            eeg, images, labels = eeg.to(device), images.to(device), labels.to(device)
            
            image_features = img_encoder.encode_image(images)
            eeg_features = eeg_encoder(eeg)
            optimizer.zero_grad()
            loss_cosine = cosine_similarity_loss(image_features, eeg_features)
            loss_cosine.backward()
            optimizer.step()
            
            train_loss += loss_cosine.item()
            train_cosine_sim += F.cosine_similarity(image_features, eeg_features).mean().item()
        
        
        train_loss /= len(train_loader)
        train_cosine_sim /= len(train_loader)
        scheduler.step()
        eeg_encoder.eval()
        img_encoder.eval()
        with torch.no_grad():
            for eeg, images, labels in tqdm(val_loader, desc="Validation"):
                eeg, images, labels = eeg.to(device), images.to(device), labels.to(device)
                
                image_features = img_encoder.encode_image(images)
                eeg_features = eeg_encoder(eeg)
                
                loss_cosine = cosine_similarity_loss(image_features, eeg_features)
                val_loss += loss_cosine.item()
                val_cosine_sim += F.cosine_similarity(image_features, eeg_features).mean().item()
        
        val_loss /= len(val_loader)
        val_cosine_sim /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {train_loss:.4f} | train cosine sim: {train_cosine_sim:.4f} | val loss: {val_loss:.4f} | val cosine sim: {val_cosine_sim:.4f}")
        
        save_fc_parameters(eeg_encoder, os.path.join(logdir, "fc_parameters.pt"))

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "train_cosine_sim": train_cosine_sim, "val_loss": val_loss, "val_cosine_sim": val_cosine_sim})
        
        if val_cosine_sim > best_cosine_sim:
            cprint("New best.", "cyan")
            torch.save({
                'fc': eeg_encoder.fc.state_dict(),  # FC層のパラメータのみを保存
                'optimizer': optimizer.state_dict(),
            }, os.path.join(logdir, "encoder_fc_best.pt"))
            best_cosine_sim = val_cosine_sim

if __name__ == "__main__":
    run()
