from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# 加载预训练模型
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# 加载数据集
# 这里，我们使用两个任务的虚拟数据：句子相似度任务（task_1_examples）和自然语言推理任务（task_2_examples）
# 您需要替换为您的实际数据集
task_1_examples = [
    InputExample(texts=['This is a sample sentence.', 'This is another sample sentence.'], label=0.8),
    InputExample(texts=['This is the third sample sentence.', 'This is the fourth sample sentence.'], label=0.9)
]

task_2_examples = [
    InputExample(texts=['All dogs are mammals.', 'All mammals are dogs.'], label=0),
    InputExample(texts=['Cats are not dogs.', 'Dogs are not cats.'], label=1)
]

# 定义自定义的collate_fn函数
def collate_fn(batch):
    texts = []
    labels = []
    for example in batch:
        texts.append(example.texts)
        labels.append(example.label)
    return {"texts": texts, "label": torch.tensor(labels, dtype=torch.float32)}

# 创建数据加载器
batch_size = 16
task_1_dataloader = DataLoader(task_1_examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
task_2_dataloader = DataLoader(task_2_examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

# 设置损失函数
task_1_loss = losses.CosineSimilarityLoss(model)
task_2_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3)  # 假设任务2具有3个类别

# 微调模型
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for task_1_batch, task_2_batch in zip(task_1_dataloader, task_2_dataloader):
        # 句子相似度任务
        model.zero_grad()
        task_1_sentences1, task_1_sentences2 = zip(*task_1_batch['texts'])
        task_1_sentences1 = list(task_1_sentences1)
        task_1_sentences2 = list(task_1_sentences2)
        task_1_labels = task_1_batch['label'].to(device)

        # 获取句子嵌入
        sentence_embeddings1 = model.encode(task_1_sentences1, convert_to_tensor=True, device=device)
        sentence_embeddings2 = model.encode(task_1_sentences2, convert_to_tensor=True, device=device)

        # 计算损失值
        task_1_loss_value = task_1_loss(sentence_embeddings1, sentence_embeddings2)
        task_1_loss_value.backward()
        # 自然语言推理任务
        task_2_sentences1, task_2_sentences2 = zip(*task_2_batch['texts'])
        task_2_labels = task_2_batch['label'].to(device)
        task_2_loss_value = task_2_loss(task_2_sentences1, task_2_sentences2, task_2_labels)
        task_2_loss_value.backward()
        # 更新权重
        model.step()
        
# 保存微调后的模型
model.save("fine-tuned-model")