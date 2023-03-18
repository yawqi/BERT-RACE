import transformers
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, InputExample, InputFeatures
from torch.utils.data import DataLoader


tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased"
        )
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.fc(outputs[1])
        return logits

def preprocess(text_a, text_b, max_len):
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_len,
        truncation_strategy="longest_first",
    )
    return (
        torch.tensor(inputs["input_ids"]),
        torch.tensor(inputs["attention_mask"]),
        torch.tensor(inputs["token_type_ids"]),
    )

import torch.optim as optim
import torch.nn.functional as F

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data_a, data_b, target) in enumerate(train_loader):
        data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_a, data_b, None)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_a, data_b, target in test_loader:
            data_a, data_b, target = (
                data_a.to(device),
                data_b.to(device),
                target.to(device),
            )
            output = model(data_a, data_b, None)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy


def main():
    # 定义训练集
    train_data = [("sentence1", "sentence2", 1), ("sentence3", "sentence4", 0)]
    train_examples = [InputExample(guid=None, text_a=t[0], text_b=t[1], label=t[2]) for t in train_data]
    train_features = [InputFeatures.from_examples(ex, max_length=128) for ex in train_examples]
    train_tensor_dataset = torch.utils.data.TensorDataset(
        torch.tensor([f.input_ids for f in train_features]),
        torch.tensor([f.attention_mask for f in train_features]),
        torch.tensor([f.token_type_ids for f in train_features]),
        torch.tensor([f.label for f in train_features]),
    )
    train_loader = DataLoader(dataset=train_tensor_dataset, batch_size=1, shuffle=True)

    # 定义测试集
    test_data = [("sentence5", "sentence6", 1), ("sentence7", "sentence8", 0)]
    test_examples = [InputExample(guid=None, text_a=t[0], text_b=t[1], label=t[2]) for t in test_data]
    test_features = [InputFeatures.from_examples(ex, max_length=128) for ex in test_examples]
    test_tensor_dataset = torch.utils.data.TensorDataset(
        torch.tensor([f.input_ids for f in test_features]),
        torch.tensor([f.attention_mask for f in test_features]),
        torch.tensor([f.token_type_ids for f in test_features]),
        torch.tensor([f.label for f in test_features]),
    )
    test_loader = DataLoader(dataset=test_tensor_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BertClassifier(n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}: validation loss: {test_loss:.4f}, accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()