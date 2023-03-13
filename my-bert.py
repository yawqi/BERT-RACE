import transformers
import torch
import json
import os
import glob
import re

from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_dir = './RACE-SR'
dev_dir = os.path.join(data_dir, 'dev')
test_dir = os.path.join(data_dir, 'test')
train_dir = os.path.join(data_dir, 'train')

def preprocess(text_a, text_b, label, max_len=512):
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_token_type_ids=True,
        return_attention_mask=True
    )
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'token_type_ids': inputs['token_type_ids'],
        'label': label
    }

def train(model, train_dataloader, optimizer, device, epochs):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        print(f'************** Epoch {epoch+1}:')
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            if step % 10 == 0:
                print("running step %d" % step)
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

def evaluate(model, test_dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            logits = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(logits.cpu().numpy())
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    print(f'Test set accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}')

def read_data_from_path(path):
    dirs = glob.glob(path+"/C*")
    data = {}
    for d in dirs:
        print("*** Scanning dir %s ***" % d)
        match = re.search(r'C(\d+)', d)
        label = int(match.group(1))
        data[label] = []
        filenames = glob.glob(d + '/*')
        for filename in filenames:
            print("Open file %s" % filename)
            with open(filename, 'r', encoding='utf-8') as fpr:
                pid = int(match.group(1))
                data_raw = json.load(fpr)
                data[label].append([data_raw['s1'], data_raw['s2']])
    return data

def main():
    # self defined parameters
    max_len = 256
    batch_size = 128
    lr = 2e-5
    epochs = 2
    device_name = "cuda:0"
    num_labels = 6

    train_data_dict = read_data_from_path(dev_dir)
    test_data_dict = read_data_from_path(test_dir)
    train_data = []
    test_data = []
    for k, v in train_data_dict.items():
        for [s1, s2] in v:
            train_data.append((s1, s2, k-1))
    
    for k, v in test_data_dict.items():
        for [s1, s2] in v:
            test_data.append((s1, s2, k-1))

    train_features = [preprocess(text_a, text_b, label, max_len) for text_a, text_b, label in train_data]
    train_dataset = TensorDataset(
        torch.tensor([f['input_ids'] for f in train_features]),
        torch.tensor([f['attention_mask'] for f in train_features]),
        torch.tensor([f['token_type_ids'] for f in train_features]),
        torch.tensor([f['label'] for f in train_features])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_features = [preprocess(text_a, text_b, label, max_len) for text_a, text_b, label in test_data]
    test_dataset = TensorDataset(
        torch.tensor([f['input_ids'] for f in test_features]),
        torch.tensor([f['attention_mask'] for f in test_features]),
        torch.tensor([f['token_type_ids'] for f in test_features]),
        torch.tensor([f['label'] for f in test_features])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    optimizer = transformers.AdamW(model.parameters(), lr=lr)

    device = torch.device(device_name)
    print("*************** Start Trainning *************")
    train(model, train_dataloader, optimizer, device, epochs)
    # Testing
    print("*************** Start Testing *************")
    evaluate(model, test_dataloader, device)

    # Save the model
    info = {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'batch_size': 8,
        'epochs': epochs
    }
    model_info_path = 'wq_model_info.json'
    model_path = 'WQBertForSequenceClassification.pth'
    torch.save(model, model_path)
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f)
if __name__ == '__main__':
    main()