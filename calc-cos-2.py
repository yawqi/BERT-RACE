import os
import json
import torch
from transformers.models.bert import BertTokenizer
import logging
import datetime
import argparse
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoLocator

from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity
# 加载预训练的BERT模型和对应的tokenizer

def calc_percent_avg(percent, list):
    start = int(len(list) * percent)
    end = int(len(list) * (1 - percent))
    avg = sum(list[start:end]) / float(len(list[start:end]))
    return start, end, avg

def read_data_from_path(path):
    s1_list = []
    s2_list = []
    filenames = glob.glob(path + '/*txt')
    for filename in filenames:
        print("Open file %s" % filename)
        with open(filename, 'r', encoding='utf-8') as fpr:
            # pid = int(match.group(1))
            data_raw = json.load(fpr)
            s1_list.append(data_raw['s1'])
            s2_list.append(data_raw['s2'])
    return s1_list, s2_list

def get_embeddings(model, text, device, max_length=512):
    # 对输入文本进行tokenization
    input_tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_tokens.to(device)
    # 使用BERT模型获取词嵌入
    with torch.no_grad():
        outputs = model(**input_tokens)
        embeddings = outputs.last_hidden_state
    # # 计算整个句子的嵌入，即各个token嵌入的平均值
    sentence_embedding = torch.mean(embeddings, dim=1)
    return sentence_embedding
    # # 获取[CLS]标记的向量作为句子的全局表示
    # cls_embedding = embeddings[:, 0, :]
    # return cls_embedding
    # 获取每个词嵌入的最大值组成的向量作为句子的全局表示
    # max_embeddings, _ = torch.max(embeddings, dim=1)
    # return max_embeddings

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    filename = "calc-cos-2-all-test.log",)

def set_args():
    parser = argparse.ArgumentParser('--CoSENT进行相似性判断')
    parser.add_argument('--pretrained_model_path', default='output-3-(1->2->3)/2023-03-25-05-21-35-original-c5', type=str, help='预训练模型的路径')
    parser.add_argument('--device', default='cpu', type=str, help='device name "cuda:#"')
    return parser.parse_args()

if __name__ == '__main__':
    args = set_args()
    # 创建输出目录
    # 加载数据集
    # model_name_or_path = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    model = BertModel.from_pretrained(args.pretrained_model_path)
    tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
    model_base = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device(args.device)
    model.to(device)
    model_base.to(device)

    max2_path = './TASK-3-DATA/test/C2'
    min2_path = './TASK-3-DATA/test/C3'
    s1_c1, s2_c1 = read_data_from_path(max2_path)
    s1_c2, s2_c2 = read_data_from_path(min2_path)
    PQs, As, Ds = [],[],[]
    sims_min, sims_max = 0.0, 0.0
    sims_base_min, sims_base_max = 0.0, 0.0
    sims_max_max, sims_min_min = 0.0, 1.0
    sims_base_max_max, sims_base_min_min = 0.0, 1.0
    sims_min_list, sims_max_list, sims_min_base_list, sims_max_base_list = [], [], [], []
    logging.info(f"Calc max cosine")
    for s1, s2 in zip(s1_c1, s2_c1):
        s1_e = get_embeddings(model, s1, device)
        s2_e = get_embeddings(model, s2, device)
        s1_base_e = get_embeddings(model_base, s1, device)
        s2_base_e = get_embeddings(model_base, s2, device)

        cos1 = cosine_similarity(s1_e, s2_e).item()
        sims_max += cos1
        cos2 = cosine_similarity(s1_base_e, s2_base_e).item()
        sims_base_max += cos2
        if cos1 > sims_max_max:
            sims_max_max = cos1
        if cos2 > sims_base_max_max:
            sims_base_max_max = cos2
        sims_max_list.append(cos1)
        sims_max_base_list.append(cos2)
        logging.info(f"Cos1: {cos1}, Cos2: {cos2}, sims_max: {sims_max}, sims_base_max: {sims_base_max}")

    sims_max_avg = sims_max / float(len(s1_c1))
    sims_base_max_avg = sims_base_max / float(len(s1_c1))
    print(f"Multitask model max2: {sims_max_avg}")
    print(f"Base model max2: {sims_base_max_avg}")
    logging.info(f"Multitask model max2: {sims_max_avg}")
    logging.info(f"Base model max2: {sims_base_max_avg}")

    logging.info(f"Calc min cosine")
    for s1, s2 in zip(s1_c2, s2_c2):
        s1_e = get_embeddings(model, s1, device)
        s2_e = get_embeddings(model, s2, device)
        s1_base_e = get_embeddings(model_base, s1, device)
        s2_base_e = get_embeddings(model_base, s2, device)

        cos1 = cosine_similarity(s1_e, s2_e).item()
        sims_min += cos1
        cos2 = cosine_similarity(s1_base_e, s2_base_e).item()
        sims_base_min += cos2
        if cos1 < sims_min_min:
            sims_min_min = cos1
        if cos2 < sims_base_min_min:
            sims_base_min_min = cos2
        sims_min_list.append(cos1)
        sims_min_base_list.append(cos2)
        logging.info(f"Cos1: {cos1}, Cos2: {cos2}, sims_min: {sims_min}, sims_base_min: {sims_base_min}")

    sims_min_avg = sims_min / float(len(s1_c2))
    sims_base_min_avg = sims_base_min / float(len(s1_c2))
    sims_min_list.sort()
    sims_max_list.sort()
    sims_min_base_list.sort()
    sims_max_base_list.sort()
    
    percent = 0.05
    start1, end1, sims_min_pavg = calc_percent_avg(percent, sims_min_list)
    start2, end2, sims_max_pavg = calc_percent_avg(percent, sims_max_list)
    start3, end3, sims_min_base_pavg = calc_percent_avg(percent, sims_min_base_list)
    start4, end4, sims_max_base_pavg = calc_percent_avg(percent, sims_max_base_list)

    print(f"Multitask model min2: {sims_min_avg} max2: {sims_max_avg}, min min2: {sims_min_min}, max max2: {sims_max_max}")
    print(f"\tremove highest and lowest {percent}: min2 avg: {sims_min_pavg} max2 avg: {sims_max_pavg}")
    print(f"\tremove highest and lowest {percent}: min2 min: {sims_min_list[start1]} max2 max: {sims_max_list[end2]}")
    print(f"Base model min2: {sims_base_min_avg} max2: {sims_base_max_avg}, min min2: {sims_base_min_min}, max max2: {sims_base_max_max}")
    print(f"\tremove highest and lowest {percent}: min2 avg: {sims_min_base_pavg} max2 avg: {sims_max_base_pavg}")
    print(f"\tremove highest and lowest {percent}: min2 min: {sims_min_base_list[start3]} max2 max: {sims_max_base_list[end4]}")

    logging.info(f"Multitask model min2: {sims_min_avg} max2: {sims_max_avg}, min min2: {sims_min_min}, max max2: {sims_max_max}")
    logging.info(f"\tremove highest and lowest {percent}: min2 avg: {sims_min_pavg} max2 avg: {sims_max_pavg}")
    logging.info(f"\tremove highest and lowest {percent}: min2 min: {sims_min_list[start1]} max2 max: {sims_max_list[end2]}")
    logging.info(f"Base model min2: {sims_base_min_avg} max2: {sims_base_max_avg}, min min2: {sims_base_min_min}, max max2: {sims_base_max_max}")
    logging.info(f"\tremove highest and lowest {percent}: min2 avg: {sims_min_base_pavg} max2 avg: {sims_max_base_pavg}")
    logging.info(f"\tremove highest and lowest {percent}: min2 min: {sims_min_base_list[start3]} max2 max: {sims_max_base_list[end4]}")

    bins = np.arange(-0.1, 1.05, 0.05)
    hist, _ = np.histogram(sims_min_list, bins=bins)

    plt.bar(bins[:-1], hist, width=0.03, align='edge')
    plt.xlabel('Value Range')
    plt.ylabel('Data Count')
    plt.title('Data Distribution')
    plt.xticks(bins + 0.05, rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(AutoLocator())
    total = sum(hist)
    print("2 sims min list:")
    logging.info("2 sims min list:")
    for i, v in enumerate(hist):
        plt.text(bins[i] + 0.04, v, str(v) + ' ({:.1f}%)'.format(v/total*100), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
        print(f"\tbin {bins[i]}: {v}")
        logging.info(f"\tbin {bins[i]}: {v}")
    plt.savefig('2_sims_min_list.png')

    hist, _ = np.histogram(sims_max_list, bins=bins)

    plt.bar(bins[:-1], hist, width=0.03, align='edge')
    plt.xlabel('Value Range')
    plt.ylabel('Data Count')
    plt.title('Data Distribution')
    plt.xticks(bins + 0.05, rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(AutoLocator())
    total = sum(hist)
    print("2 sims max list:")
    logging.info("2 sims max list:")
    for i, v in enumerate(hist):
        plt.text(bins[i] + 0.04, v, str(v) + ' ({:.1f}%)'.format(v/total*100), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
        print(f"\tbin {bins[i]}: {v}")
        logging.info(f"\tbin {bins[i]}: {v}")
    plt.savefig('2_sims_max_list.png')

    hist, _ = np.histogram(sims_min_base_list, bins=bins)

    plt.bar(bins[:-1], hist, width=0.03, align='edge')
    plt.xlabel('Value Range')
    plt.ylabel('Data Count')
    plt.title('Data Distribution')
    plt.xticks(bins + 0.05, rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(AutoLocator())
    total = sum(hist)
    print("2 sims min base list:")
    logging.info("2 sims min base list:")
    for i, v in enumerate(hist):
        plt.text(bins[i] + 0.04, v, str(v) + ' ({:.1f}%)'.format(v/total*100), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
        print(f"\tbin {bins[i]}: {v}")
        logging.info(f"\tbin {bins[i]}: {v}")

    plt.savefig('2_sims_min_base_list.png')

    hist, _ = np.histogram(sims_max_base_list, bins=bins)

    plt.bar(bins[:-1], hist, width=0.03, align='edge')
    plt.xlabel('Value Range')
    plt.ylabel('Data Count')
    plt.title('Data Distribution')
    plt.xticks(bins + 0.05, rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(AutoLocator())
    total = sum(hist)
    print("2 sims max base list:")
    logging.info("2 sims max base list:")
    for i, v in enumerate(hist):
        plt.text(bins[i] + 0.04, v, str(v) + ' ({:.1f}%)'.format(v/total*100), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
        print(f"\tbin {bins[i]}: {v}")
        logging.info(f"\tbin {bins[i]}: {v}")
    plt.savefig('2_sims_max_base_list.png')