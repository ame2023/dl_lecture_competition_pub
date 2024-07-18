import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
import numpy as np
from src.utils import set_seed
from pretrained_MAE2 import MAE, PatchEmbedding  # MAEとPatchEmbeddingを適当なディレクトリからインポート
from CLIP_preEEG import ModifiedMAE

# BERTモデルのロード
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

def initialize_weights(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# 前処理済みデータをロードするデータセットクラス
class PreprocessedEEGImageDataset(Dataset):
    def __init__(self, eeg_data_file, labels_file=None):
        self.eeg_data = torch.load(eeg_data_file)
        if labels_file:
            self.labels = torch.load(labels_file)
        else:
            self.labels = None

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        return eeg, self.labels[idx] if self.labels is not None else None

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="EEG-classification")

    device = args.device

    # データロード
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_train_X.pt'
    train_labels_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_train_y.pt'
    val_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_val_X.pt'
    val_labels_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_val_y.pt'
    test_eeg_data_file = r'C:\Users\controllab\dl_lecture_competition_pub\data\preprocessed_test_X.pt'

    train_dataset = PreprocessedEEGImageDataset(train_eeg_data_file, train_labels_file)
    val_dataset = PreprocessedEEGImageDataset(val_eeg_data_file, val_labels_file)
    test_dataset = PreprocessedEEGImageDataset(test_eeg_data_file)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    # モデル定義
    num_classes = len(torch.unique(torch.load(train_labels_file)))
    classifier = BertClassifier(bert_model, input_dim=768, num_classes=num_classes).to(device)

    # 事前学習したEEGエンコーダとfc層のパラメータをロード
    mae_checkpoint_path = r'C:\Users\controllab\dl_lecture_competition_pub\model_best.pt'
    fc_checkpoint_path = r'C:\Users\controllab\dl_lecture_competition_pub\fc_parameters.pt'

    mae_model = ModifiedMAE(input_dim=271, embed_dim=256, num_layers=6, num_heads=8, hidden_dim=512, dropout_rate=0.2, output_dim=512)
    mae_model.mae.encoder.load_state_dict(torch.load(mae_checkpoint_path)['eeg_encoder'])
    mae_model.fc.load_state_dict(torch.load(fc_checkpoint_path)['fc'])

    mae_model.to(device)
    mae_model.eval()  # MAEモデルのパラメータを固定
    for param in mae_model.parameters():
        param.requires_grad = False

    # オプティマイザ定義
    optimizer = torch.optim.AdamW([
        {'params': classifier.parameters(), 'lr': args.lr, 'weight_decay': 5e-2},
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)

    best_loss = float('inf')
    best_acc = float('-inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, val_loss = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0
        
        classifier.train()
        for eeg, labels in tqdm(train_loader, desc="Train"):
            eeg, labels = eeg.to(device), labels.to(device)
            
            with torch.no_grad():
                eeg_features = mae_model(eeg)
            
            outputs = classifier(eeg_features)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy_metric(outputs, labels).item()
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        classifier.eval()
        with torch.no_grad():
            for eeg, labels in tqdm(val_loader, desc="Validation"):
                eeg, labels = eeg.to(device), labels.to(device)
                
                with torch.no_grad():
                    eeg_features = mae_model(eeg)
                
                outputs = classifier(eeg_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += accuracy_metric(outputs, labels).item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")
        
        torch.save({
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(logdir, "classifier_last.pt"))

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        
        if val_loss < best_loss:
            cprint("New best.", "cyan")
            torch.save({
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(logdir, "classifier_best.pt"))
            best_loss = val_loss

    # 最適なパラメータをロード
    best_checkpoint_path = os.path.join(logdir, "classifier_best.pt")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['classifier'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # テストデータでの予測
    preds = []
    classifier.eval()
    with torch.no_grad():
        for eeg, _ in tqdm(test_loader, desc="Test"):
            eeg = eeg.to(device)
            eeg_features = mae_model(eeg)
            outputs = classifier(eeg_features)
            preds.append(outputs.detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()
