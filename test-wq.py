from transformers import BertTokenizer, BertModel
import torch

# 加载预训练的BERT模型和对应的tokenizer
model_name_or_path = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertModel.from_pretrained(model_name_or_path)

# 输入文本
text = "This is an example sentence."

# 对输入文本进行tokenization
input_tokens = tokenizer(text, return_tensors="pt")

# 使用BERT模型获取词嵌入
with torch.no_grad():
    outputs = model(**input_tokens)
    embeddings = outputs.last_hidden_state

# `embeddings`是一个形状为(batch_size, sequence_length, hidden_size)的张量
# 其中，batch_size=1，sequence_length是输入文本的长度，hidden_size是BERT模型的隐藏层大小
print(embeddings.shape)