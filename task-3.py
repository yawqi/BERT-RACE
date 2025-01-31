import re
import os
import random
import torch
import numpy as np
import scipy
from tqdm import tqdm
from config import set_args
from model import Model
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_helper import CustomDataset, collate_fn, pad_to_maxlen, load_data, load_test_data
import logging
import datetime

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def compute_pearsonr(x, y):
    return scipy.stats.pearsonr(x, y)[0]

def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def get_sent_id_tensor(s_list):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    max_len = min(max([len(_)+2 for _ in s_list]), 512)   # 这样写不太合适 后期想办法改一下
    for s in s_list:
        inputs = tokenizer.encode_plus(text=s, text_pair=None, add_special_tokens=True, return_token_type_ids=True)
        input_ids.append(pad_to_maxlen(inputs['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(inputs['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(inputs['token_type_ids'], max_len=max_len))
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids

def evaluate(device, cur_dir):
    sent1, sent2, label = load_test_data(args.test_data, max_label=3)
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    device = torch.device(device)
    model.to(device)
    model.eval()
        # 创建输出目录
    corrcoef = 100.0

    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    log_file = os.path.join(cur_dir, "evaluation-log.txt")
    with open(log_file, "w") as log:
        log.write("Evaluation started at {}\n".format(timestamp))
        for i, (s1, s2, lab) in enumerate(tqdm(zip(sent1, sent2, label))):
            input_ids, input_mask, segment_ids = get_sent_id_tensor([s1, s2])
            lab = torch.tensor([lab], dtype=torch.float)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            lab = lab.to(device)
            # if torch.cuda.is_available():
            #     input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
            #     lab = lab.cuda()

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type=encoder_type)
            output = output.to(device)
            all_a_vecs.append(output[0].cpu().numpy())
            all_b_vecs.append(output[1].cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
            # 每100个样本打印一次信息
            if i % 100 == 0:
                log.write(f"Processed {i} samples\n")
                print(f"Processed {i} samples")

        all_a_vecs = np.array(all_a_vecs)
        all_b_vecs = np.array(all_b_vecs)
        all_labels = np.array(all_labels)

        a_vecs = l2_normalize(all_a_vecs)
        b_vecs = l2_normalize(all_b_vecs)
        sims = (a_vecs * b_vecs).sum(axis=1)

        c1_c2_labels = [a for a, b in zip(all_labels, sims) if a == 1 or a == 2]
        c1_c2_sims = [b for a, b in zip(all_labels, sims) if a == 1 or a == 2]
        
        c1_c3_labels = [a for a, b in zip(all_labels, sims) if a == 0 or a == 2]
        c1_c3_sims = [b for a, b in zip(all_labels, sims) if a == 0 or a == 2]

        c2_c3_labels = [a for a, b in zip(all_labels, sims) if a == 1 or a == 0]
        c2_c3_sims = [b for a, b in zip(all_labels, sims) if a == 1 or a == 0]
        corrcoef_all = compute_corrcoef(all_labels, sims)
        corrcoef_c1_c2 = compute_corrcoef(c1_c2_labels, c1_c2_sims)
        corrcoef_c1_c3 = compute_corrcoef(c1_c3_labels, c1_c3_sims)
        corrcoef_c2_c3 = compute_corrcoef(c2_c3_labels, c2_c3_sims)

        p_all = compute_pearsonr(all_labels, sims)
        p_c1_c2 = compute_pearsonr(c1_c2_labels, c1_c2_sims)
        p_c1_c3 = compute_pearsonr(c1_c3_labels, c1_c3_sims)
        p_c2_c3 = compute_pearsonr(c2_c3_labels, c2_c3_sims)

        log.write(f"All correlation coefficient: {corrcoef_all}\n")
        log.write(f"C1 C2 correlation coefficient: {corrcoef_c1_c2}\n")
        log.write(f"C1 C3 correlation coefficient: {corrcoef_c1_c3}\n")
        log.write(f"C2 C3 correlation coefficient: {corrcoef_c2_c3}\n")
        log.write(f"All pearson: {p_all}\n")
        log.write(f"C1 C2 pearson: {p_c1_c2}\n")
        log.write(f"C1 C3 pearson: {p_c1_c3}\n")
        log.write(f"C2 C3 pearson: {p_c2_c3}\n")

        print(f"All correlation coefficient: {corrcoef_all}")
        print(f"C1 C2 correlation coefficient: {corrcoef_c1_c2}")
        print(f"C1 C3 correlation coefficient: {corrcoef_c1_c3}")
        print(f"C2 C3 correlation coefficient: {corrcoef_c2_c3}")
        print(f"All pearson: {p_all}")
        print(f"C1 C2 pearson: {p_c1_c2}")
        print(f"C1 C3 pearson: {p_c1_c3}")
        print(f"C2 C3 pearson: {p_c2_c3}")
    return corrcoef

def calc_loss(y_true, y_pred, device):
    # 1. 取出真实的标签
    y_true = y_true[::2]    # tensor([1, 0, 1]) 真实的标签

    # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    y_pred = y_pred / norms

    # 3. 奇偶向量相乘
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20

    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]   # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        
    return torch.logsumexp(y_pred, dim=0)

if __name__ == '__main__':
    args = set_args()
    set_seed()

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cur_dir = os.path.join(args.output_dir, f"{timestamp}")
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # 加载数据集
    train_sentence, train_label = load_data(args.train_data, max_label=3)
    last_three = True
    encoder_type = 'first-last-avg'
    train_dataset = CustomDataset(sentence=train_sentence, label=train_label, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=1)

    total_steps = len(train_dataloader) * args.num_train_epochs

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = Model(pretrain_model_path_or_name=args.pretrained_model_path, froze_params=False, last_three = last_three)
    device = torch.device(args.device)
    model.to(device)
    # if torch.cuda.is_available():
    #     model.cuda(1)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    for epoch in range(args.num_train_epochs):
        model.train()
        train_label, train_predict = [], []
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            # for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            # if torch.cuda.is_available():
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type=encoder_type)
            loss = calc_loss(label_ids, output, device=args.device)
            loss.backward()
            s = "当前轮次:{}, 正在迭代:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss)
            print(s)  # 在进度条前面定义一段文字
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            log_path = os.path.join(cur_dir, 'logs.txt')
            with open(log_path, 'a+') as f:
                s += '\n'
                f.write(s)
            epoch_loss += loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        corr = evaluate(args.device, cur_dir)
        s = 'Epoch:{} | corr: {:10f}'.format(epoch, corr)
        logs_path = os.path.join(cur_dir, 'result-logs.txt')
        with open(logs_path, 'a+') as f:
            s += '\n'
            f.write(s)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(cur_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)