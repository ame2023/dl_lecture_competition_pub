import os
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
import numpy as np
from src.utils import set_seed

# CLIPモデルのロード
ViT_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')

# BERTモデルのロード
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 画像の前処理を定義
preprocess_image = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ViT_model.to('cuda')
bert_model.to('cuda')
ViT_model.eval()  # ViTのパラメータを固定
for param in ViT_model.parameters():
    param.requires_grad = False

# ViTモデルの出力を384次元に変換するための線形層を追加
class ModifiedViTModel(nn.Module):
    def __init__(self, original_model, input_dim=512, output_dim=384):
        super(ModifiedViTModel, self).__init__()
        self.original_model = original_model
        self.fc = nn.Linear(input_dim, output_dim)
        self.fc.requires_grad = True  # fc層のみを最適化対象に設定
        initialize_weights(self.fc)  # fc層のみを初期化

    def encode_image(self, image):
        with torch.no_grad():
            features = self.original_model.encode_image(image)
        return self.fc(features)


def initialize_weights(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# EEGエンコーダを定義
class TransformerEEGEncoder(nn.Module):
    def __init__(self, input_dim, num_heads=8, hidden_dim=512, num_layers=6, output_dim=384, dropout_rate=0.3, dim_feedforward=2048):
        super(TransformerEEGEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(initialize_weights)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# クラス分類用のBERT
class BertClassifier(nn.Module):
    def __init__(self, bert_model, input_dim, num_classes):
        super(BertClassifier, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(input_dim, num_classes)
        self.norm = nn.BatchNorm1d(768)
        initialize_weights(self.fc)  # fc層のみを初期化

    def forward(self, combined_features):
        combined_features = self.norm(combined_features)
        outputs = self.bert_model(inputs_embeds=combined_features.unsqueeze(1))
        logits = self.fc(outputs.pooler_output)
        return logits


# 前処理済みデータをロードするデータセットクラス
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
        if self.image_paths:
            image_path = self.valid_image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = preprocess_image(image)
            return eeg, image, self.labels[idx] if self.labels is not None else None
        else:
            return eeg

def mask_vector(vector, mask_prob=0.5, mask_value=0):
    mask = torch.bernoulli(torch.full(vector.shape, mask_prob)).to(vector.device)
    return vector * (1 - mask) + mask_value * mask

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="EEG-classification")

    device = args.device
    mask_value = 0 # マスクの値　0か-1e9

    # データロード
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
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

    # モデル定義
    num_classes = len(torch.unique(torch.load(train_labels_file)))
    eeg_encoder = TransformerEEGEncoder(input_dim=271, num_heads=8, hidden_dim=512, num_layers=6, output_dim=768, dropout_rate=0.2).to(device)
    model = ModifiedViTModel(ViT_model, input_dim=512, output_dim=768).to(device)
    classifier = BertClassifier(bert_model, input_dim=768, num_classes=num_classes).to(device)

    # オプティマイザ定義
    optimizer = torch.optim.AdamW([
        {'params': eeg_encoder.parameters(), 'lr': args.lr*0.1, 'weight_decay': 1e-2},
        {'params': classifier.parameters(), 'lr': args.lr, 'weight_decay': 5e-2},
        {'params': model.fc.parameters(), 'lr': args.lr*0.1, 'weight_decay': 1e-2},  # ViTモデルのfc層のみを最適化対象に追加
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=10).to(device)

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

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, val_loss = 0.0, 0.0
        train_cosine_sim, val_cosine_sim = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0
        
        eeg_encoder.train()
        classifier.train()
        for eeg, images, labels in tqdm(train_loader, desc="Train"):
            eeg, images, labels = eeg.to(device), images.to(device), labels.to(device)
            
            image_features = model.encode_image(images)
            eeg_features = eeg_encoder(eeg)
            #combined_features = torch.cat((eeg_features, image_features), dim=1)
            #combined_features = mask_vector(combined_features, mask_prob=0.5, mask_value=mask_value)  # マスクの値は0 or -1e9
            #outputs = classifier(combined_features)
            outputs = classifier(eeg_features)
            optimizer.zero_grad()
            loss_cosine = cosine_similarity_loss(image_features, eeg_features)
            loss_ce = criterion(outputs, labels)
            loss = loss_cosine + loss_ce
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_cosine_sim += F.cosine_similarity(image_features, eeg_features).mean().item()
            train_acc += accuracy_metric(outputs, labels).item()
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_cosine_sim /= len(train_loader)
        train_acc /= len(train_loader)
        
        eeg_encoder.eval()
        classifier.eval()
        with torch.no_grad():
            for eeg, images, labels in tqdm(val_loader, desc="Validation"):
                eeg, images, labels = eeg.to(device), images.to(device), labels.to(device)
                
                image_features = model.encode_image(images)
                eeg_features = eeg_encoder(eeg)
                #combined_features = torch.cat((eeg_features, image_features), dim=1)
                outputs = classifier(eeg_features)
                
                loss_cosine = cosine_similarity_loss(image_features, eeg_features)
                loss_ce = criterion(outputs, labels)
                loss = loss_cosine + loss_ce
                val_loss += loss.item()
                val_cosine_sim += F.cosine_similarity(image_features, eeg_features).mean().item()
                val_acc += accuracy_metric(outputs, labels).item()
        
        val_loss /= len(val_loader)
        val_cosine_sim /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {train_loss:.4f} | train cosine sim: {train_cosine_sim:.4f} | train acc: {train_acc:.4f} | val loss: {val_loss:.4f} | val cosine sim: {val_cosine_sim:.4f} | val acc: {val_acc:.4f}")
        
        torch.save({
            'eeg_encoder': eeg_encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'fc': model.fc.state_dict(),  # fc層のパラメータを保存
        }, os.path.join(logdir, "model_last.pt"))

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "train_cosine_sim": train_cosine_sim, "train_acc": train_acc, "val_loss": val_loss, "val_cosine_sim": val_cosine_sim, "val_acc": val_acc})
        
        if val_loss < best_loss:
            cprint("New best.", "cyan")
            torch.save({
                'eeg_encoder': eeg_encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'fc': model.fc.state_dict(),  # fc層のパラメータを保存
            }, os.path.join(logdir, "model_best.pt"))
            best_loss = val_loss
    
    # テストデータでの予測
    checkpoint = torch.load(os.path.join(logdir, "model_best.pt"), map_location=device)
    eeg_encoder.load_state_dict(checkpoint['eeg_encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    model.fc.load_state_dict(checkpoint['fc'])  # fc層のパラメータをロード
    optimizer.load_state_dict(checkpoint['optimizer'])

    preds = []
    eeg_encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for eeg in tqdm(test_loader, desc="Test"):
            eeg = eeg.to(device)
            eeg_features = eeg_encoder(eeg)
            #dummy_image_features = torch.full_like(eeg_features, mask_value)  # マスク値を使用
            #combined_features = torch.cat((eeg_features, dummy_image_features), dim=1)
            #outputs = classifier(combined_features)
            outputs = classifier(eeg_features)
            preds.append(outputs.detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()
