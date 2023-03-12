import transformers
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.fc(outputs[1])
        return logits
    
def preprocess(text_a, text_b, max_len):
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_len,
        truncation_strategy='longest_first'
    )
    return torch.tensor(inputs['input_ids']), torch.tensor(inputs['attention_mask']), torch.tensor(inputs['token_type_ids'])

import torch.optim as optim
import torch.nn.functional as F

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data_a, data_b, target) in enumerate(train_loader):
        data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_a, data_b)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_a, data_b, target in test_loader:
            data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
            output = model(data_a, data_b)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def main():
    train_data = [('sentence1', 'sentence2', 1), ('sentence1', 'sentence2', 0), ...]
    test_data = [('sentence1', 'sentence2', 1), ('sentence1', 'sentence2', 0), ...]
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, collate_fn=lambda x: (torch.stack([i[0] for i in x]), torch.stack([i[1] for i in x]), torch.tensor([i[2] for i in x])))
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, collate_fn=lambda x: (torch.stack([i[0] for i in x]), torch.stack([i[1] for i in x]), torch.tensor([i[2] for i in x])))

    torch.cuda.set_device(0)
    device = torch.device("cuda:1")
    model = BertClassifier(n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = F.cross_entropy
    for epoch in range(10):
        train(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        print(f'{epoch:d}validation loss: {test_loss:.4f}, accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()