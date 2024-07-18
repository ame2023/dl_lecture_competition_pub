import re
import random
import time
from statistics import mode
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import BertTokenizer, BertModel, CLIPModel, CLIPProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, processor, transform=None, answer=True):
        self.transform = transform
        self.processor = processor
        self.image_dir = image_dir
        self.df = df
        self.answer = answer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.class_mapping = self.load_class_mapping()
        if self.answer:
            self.answer2idx = {answer: idx for idx, answer in enumerate(self.class_mapping)}
            self.idx2answer = {idx: answer for answer, idx in self.answer2idx.items()}

    def load_class_mapping(self):
        class_mapping = pd.read_csv("class_mapping.csv")
        return class_mapping['answer'].tolist()

    def update_dict(self, dataset):
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df.iloc[idx]['image']}")
        if self.transform:
            image = self.transform(image)
        image = self.processor(images=image, return_tensors="pt", do_rescale=False)["pixel_values"].squeeze()
        question = self.df.iloc[idx]["question"]
        inputs = self.tokenizer(question, return_tensors="pt", padding='max_length', truncation=True, max_length=32)
        if self.answer:
            answers = [self.answer2idx.get(process_text(answer["answer"]), 0) for answer in self.df.iloc[idx]["answers"]]
            mode_answer_idx = mode(answers)
            return image, inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

    def __len__(self):
        return len(self.df)

def VQA_criterion(batch_pred, batch_answers):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

class VQAModel(nn.Module):
    def __init__(self, n_answer):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )
        
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, image, question, attention_mask):
        image_feature = self.clip.get_image_features(image)
        question_feature = self.bert(question, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, attention_mask, answers, mode_answer in tqdm(dataloader, desc="Training"):
        image, question, attention_mask, answers, mode_answer = image.to(device), question.to(device), attention_mask.to(device), answers.to(device), mode_answer.to(device)
        batch_loss = 0
        batch_acc = 0
        batch_simple_acc = 0
        for i in range(len(image)):
            img = image[i].unsqueeze(0)
            ques = question[i].unsqueeze(0)
            att_mask = attention_mask[i].unsqueeze(0)
            ans = answers[i].unsqueeze(0)
            mode_ans = mode_answer[i].item()  
            
            if mode_ans < 0 or mode_ans >= model.fc[-1].out_features:
                continue
            pred = model(img, ques, att_mask)
            loss = criterion(pred, torch.tensor([mode_ans], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            batch_acc += VQA_criterion(pred.argmax(1), ans)
            batch_simple_acc += (pred.argmax(1) == mode_ans).float().mean().item()
        total_loss += batch_loss / len(image)
        total_acc += batch_acc / len(image)
        simple_acc += batch_simple_acc / len(image)
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 5:
                image, question, attention_mask, answers, mode_answer = batch
                image, question, attention_mask, answers, mode_answer = image.to(device), question.to(device), attention_mask.to(device), answers.to(device), mode_answer.to(device)
            else:
                image, question, attention_mask = batch
                image, question, attention_mask = image.to(device), question.to(device), attention_mask.to(device)
                answers, mode_answer = None, None

            batch_loss = 0
            batch_acc = 0
            batch_simple_acc = 0
            for i in range(len(image)):
                img = image[i].unsqueeze(0)
                ques = question[i].unsqueeze(0)
                att_mask = attention_mask[i].unsqueeze(0)
                if answers is not None and mode_answer is not None:
                    ans = answers[i].unsqueeze(0)
                    mode_ans = mode_answer[i].item()  
                    
                    if mode_ans < 0 or mode_ans >= model.fc[-1].out_features:
                        continue
                    pred = model(img, ques, att_mask)
                    loss = criterion(pred, torch.tensor([mode_ans], device=device))
                    batch_loss += loss.item()
                    batch_acc += VQA_criterion(pred.argmax(1), ans)
                    batch_simple_acc += (pred.argmax(1) == mode_ans).float().mean().item()
                else:
                    pred = model(img, ques, att_mask)
                    loss = 0  
                    batch_loss += loss
            total_loss += batch_loss / len(image)
            total_acc += batch_acc / len(image)
            simple_acc += batch_simple_acc / len(image)
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    df = pd.read_json("./data/train.json")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    train_dataset = VQADataset(df=train_df, image_dir="./data/train", processor=processor, transform=transform)
    val_dataset = VQADataset(df=val_df, image_dir="./data/train", processor=processor, transform=None)
    val_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_loss = float('inf')
    best_model_wts = None
    num_epoch = 50

    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_simple_acc, val_time = eval(model, val_loader, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"val loss: {val_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"val acc: {val_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}\n"
              f"val simple acc: {val_simple_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, "train_best_params.pth")

        torch.save(model.state_dict(), "train_last_epo_params.pth")

        scheduler.step()

    model.load_state_dict(best_model_wts)
    model.eval()
    submission = []
    for image, question, attention_mask in val_loader:
        image, question, attention_mask = image.to(device), question.to(device), attention_mask.to(device)
        pred = model(image, question, attention_mask)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)

    print(f"Submission: {submission[:10]}")  

    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission, allow_pickle=True)  

if __name__ == "__main__":
    main()
