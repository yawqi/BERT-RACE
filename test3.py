from transformers import InputExample, InputFeatures
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def preprocess(text_a, text_b, label, max_len):
    inputs = tokenizer.encode_plus(
        text_a, 
        text_b, 
        add_special_tokens=True,
        max_length=max_len, 
        truncation_strategy='longest_first'
    )
    return InputFeatures(
        input_ids=torch.tensor(inputs['input_ids'], dtype=torch.long),
        token_type_ids=None, # 如果只有一个句子，bert 默认将其视为第一个句子，第二个句子的 token_type_ids 全部设置为1
        attention_mask=torch.tensor(inputs['attention_mask'], dtype=torch.long),
        label=torch.tensor(label, dtype=torch.long)
    )

def train(model, train_dataloader, optimizer, device, epochs):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[3].to(device)

            model.zero_grad()   
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

def main():
    train_data = [('I love ice cream!', 'I hate chocolate ice cream!', 0), ('You are so cute!', 'You are so boring!', 1)]
    max_len = 50
    train_features = [preprocess(text_a, text_b, label, max_len) for text_a, text_b, label in train_data]
    train_dataset = TensorDataset(
        torch.stack([f.input_ids for f in train_features]),
        torch.stack([f.attention_mask for f in train_features]),
        torch.stack([f.label for f in train_features])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    epochs = 2
    lr = 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_dataloader, optimizer, device, epochs)

if __name__ == '__main__':
    main()