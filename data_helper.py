import torch
from torch.utils.data import Dataset
import glob
import json
import re

# def load_data(path):
#     sentence, label = [], []
#     with open(path, 'r', encoding='utf8') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip().split('\t')
#             try:
#                 sentence.extend([line[0], line[1]])
#                 lab = int(line[2])
#                 label.extend([lab, lab])
#             except:
#                 continue
#     return sentence, label

def load_data(path, max_label = None):
    dirs = glob.glob(path+"/C*")
    sentences = []
    labels = []
    f_names = []
    count = 0
    sorted(dirs)
    if not max_label:
        max_label = len(dirs)

    for i in range(0, max_label):
        filenames = glob.glob(dirs[i] + '/*')
        f_names.append(filenames)
        if count == 0:
            count = len(filenames)
        else:
            count = min(count, len(filenames))

    for i in range(0, count):
        for j in range(0, max_label):
                filename = f_names[j][i]

                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    sentences.extend([data_raw['s1'], data_raw['s2']])
                    labels.extend([max_label - 1 - j, max_label - 1 - j])
    return sentences, labels

def load_test_data(path, max_label = None):
    dirs = glob.glob(path+"/C*")
    sentences1 = []
    sentences2 = []
    labels = []
    for d in dirs:
        match = re.search(r'C(\d+)', d)
        label = int(match.group(1)) - 1
        if max_label and max_label - 1 < label:
            continue
        label = max_label - 1 - label
        filenames = glob.glob(d + '/*')
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                sentences1.append(data_raw['s1'])
                sentences2.append(data_raw['s2'])
                labels.append(label)
    return sentences1, sentences2, labels

# def load_test_data(path):
#     sent1, sent2, label = [], [], []
#     with open(path, 'r', encoding='utf8') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip().split('\t')
#             sent1.append(line[0])
#             sent2.append(line[1])
#             label.append(int(line[2]))
#     return sent1, sent2, label


class CustomDataset(Dataset):
    def __init__(self, sentence, label, tokenizer):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len = max([len(d['input_ids']) for d in batch])

    if max_len > 512:
        max_len = 512
    # 定一个全局的max_len
    # max_len = 128

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids