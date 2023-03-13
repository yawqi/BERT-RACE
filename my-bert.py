import transformers
import torch
import json
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
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
        print(f'Epoch {epoch+1}:')
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
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

def main():
    train_data = [("I love ice cream!", "I hate chocolate ice cream!", 0), ("You are so cute!", "You are so boring!", 1)]
    max_len = 50
    train_features = [preprocess(text_a, text_b, label, max_len) for text_a, text_b, label in train_data]
    train_dataset = TensorDataset(
        torch.tensor([f['input_ids'] for f in train_features]),
        torch.tensor([f['attention_mask'] for f in train_features]),
        torch.tensor([f['token_type_ids'] for f in train_features]),
        torch.tensor([f['label'] for f in train_features])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = transformers.AdamW(model.parameters(), lr=2e-5)

    epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_dataloader, optimizer, device, epochs)

    test_data = [("I hate ice cream!", "I love chocolate ice cream!", 0), ("He is very handsome.", "She is very beautiful.", 1)]
    test_features = [preprocess(text_a, text_b, label, max_len) for text_a, text_b, label in test_data]
    test_dataset = TensorDataset(
        torch.tensor([f['input_ids'] for f in test_features]),
        torch.tensor([f['attention_mask'] for f in test_features]),
        torch.tensor([f['token_type_ids'] for f in test_features]),
        torch.tensor([f['label'] for f in test_features])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # Testing
    evaluate(model, test_dataloader, device)

    # Save the model
    info = {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'batch_size': 8,
        'epochs': epochs
    }
    model_info_path = 'model_info.json'
    model_path = 'BertForSequenceClassification.pth'
    torch.save(model, model_path)
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f)
if __name__ == '__main__':
    main()