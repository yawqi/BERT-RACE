import glob
import os
import json
import torch
from transformers.models.bert import BertTokenizer
import logging
import datetime
import argparse
import re

from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity
# 加载预训练的BERT模型和对应的tokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from bert_score import score

def bleu_score(reference_sentence, candidate_sentence):
    reference_tokens = word_tokenize(reference_sentence.lower())
    candidate_tokens = word_tokenize(candidate_sentence.lower())

    # 由于我们只有一个参考句子，我们需要将其放在一个列表中
    reference = [reference_tokens]
    return sentence_bleu(reference, candidate_tokens)

# def get_embeddings(model, text, device):
#     # 对输入文本进行tokenization
#     input_tokens = tokenizer(text, return_tensors="pt")
#     input_tokens.to(device)
#     # 使用BERT模型获取词嵌入
#     with torch.no_grad():
#         outputs = model(**input_tokens)
#         embeddings = outputs.last_hidden_state

#     # 计算整个句子的嵌入，即各个token嵌入的平均值
#     sentence_embedding = torch.mean(embeddings, dim=1)
#     return sentence_embedding

def get_embeddings(model, text, device):
    # 对输入文本进行tokenization
    input_tokens = tokenizer(text, return_tensors="pt")
    input_tokens.to(device)
    # 使用BERT模型获取词嵌入
    with torch.no_grad():
        outputs = model(**input_tokens)
        embeddings = outputs.last_hidden_state
    # 计算整个句子的嵌入，即各个token嵌入的平均值
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
                    filename = "exp-task-output-1-C4-new-last-three-mean-(2->3->1)-new.log",)

def set_args():
    parser = argparse.ArgumentParser('--CoSENT进行相似性判断')
    parser.add_argument('--test_data', default='./NEW-RACE-TEST', type=str, help='测试数据集')
    parser.add_argument('--pretrained_model_path', default='output-1-C4-new-last-three-mean-(2->3->1)/task-1-.-output-3-new-(2->3)-last-three-mean-2023-04-04-14-00-43--2023-04-07_00-56-27', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./exp-result', type=str, help='模型输出')
    parser.add_argument('--train_batch_size', default=32, type=int, help='训练批次大小')
    parser.add_argument('--val_batch_size', default=32, type=int, help='验证批次大小')
    parser.add_argument('--device', default='cuda:0', type=str, help='device name "cuda:#"')
    return parser.parse_args()

# def read_q_from_p(pid, path):
#     filenames = glob.glob(path+f"/{pid}-*txt")
#     qid = pid % len(filenames)
#     filename = os.path.join(path, f"{pid}-{qid}.txt")
#     pq, ans, d = None, None, None
#     with open(filename, 'r', encoding='utf-8') as fpr:
#         data_raw = json.load(fpr)
#         pq = data_raw['PQ']
#         ans = data_raw['A']
#         d = data_raw['D'][:3]
#     return pq, ans, d, qid

def read_q_from_p(pid, qid, path):
    filename = os.path.join(path, f"{pid}-{qid}.txt")
    pq, ans, d = None, None, None
    with open(filename, 'r', encoding='utf-8') as fpr:
        data_raw = json.load(fpr)
        pq = data_raw['PQ']
        ans = data_raw['A']
        if len(data_raw['D']) < 3:
            return pq, ans, d
        d = data_raw['D'][:3]
    return pq, ans, d

def regulate_score(sc):
    if sc > 1.0:
        sc = 1.0
    elif sc < 0.0:
        sc = 0.0
    return sc

def calc_score(pq_e, a_e, target_e, max1, min1, max2, min2, a_e_2 = None,  target_e_2 = None):
    sim_pq_d = cosine_similarity(pq_e, target_e).item()
    sim_ans_d = None
    if target_e_2 is None or a_e_2 is None:
        sim_ans_d = cosine_similarity(a_e, target_e).item()
    else:
        sim_ans_d = cosine_similarity(a_e_2, target_e_2).item()

    sc_pq_ans = (sim_pq_d - min2) / (max2 - min2)
    sc_ans_di = (max1 - sim_ans_d) / (max1 - min1)
    sc_avg = (sc_pq_ans + sc_ans_di) / 2.0

    return sim_pq_d, sim_ans_d, sc_pq_ans, sc_ans_di, sc_avg

# def write_to_file(i, qid, d, pq_e, d_e, a_e, pq_e_base, d_e_base, a_e_base,
# min1, max1, min1_base, max1_base, min2, max2, min2_base, max2_base):
#     with open('information.txt', 'a+', encoding='utf-8') as f:
#         f.write(f"pid: {i}, qid: {qid}\n")
#         for j in range(3):
#             f.write(f"\t d[{j}]: {d[j]}\n")
#             f.write(f"\tsc {j}, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}, pq_d: {sim_pq_d}, ans_d: {sim_ans_d}\n")
#             f.write(f"\tsc_base {j}, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}, pq_d: {sim_pq_d_base}, ans_d: {sim_ans_d_base}\n")
#             f.write("\tAfter regulate:\n")
#             f.write(f"\tsc {j}, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}\n")
#             f.write(f"\tsc_base {j}, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}\n")

if __name__ == '__main__':
    args = set_args()
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cur_dir = os.path.join(args.output_dir, f"{timestamp}")
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

    # 加载数据集
    # model_name_or_path = 'bert-base-uncased'
    path = args.test_data
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    model = BertModel.from_pretrained(args.pretrained_model_path)
    tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
    model_base = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device(args.device)
    model.to(device)
    model_base.to(device)
    # C4
    max1 = 0.9521908760070801
    min1 = 0.5939056873321533

    max1_base = 0.8787095546722412
    min1_base = 0.5403128266334534

    max2 = 0.8582397103309631
    min2 = -0.05112775042653084 
    max2_base = 0.7596578598022461
    min2_base = 0.25286248326301575

    # max1 = 0.989
    # min1 = 0.273

    # max1_base = 0.908
    # min1_base = 0.525

    # max2 = 0.846
    # min2 = -0.106
    # max2_base = 0.762
    # min2_base = 0.251
    PQs, As, Ds = [],[],[]
    filenames = glob.glob(path+f"/*-*.txt")
    filenames.sort()
    prev_d = "This is a wrong answer."
    idx = 0
    total = len(filenames)
    prev_pid = -1
    tmp_d = "This is a wrong answer."
    filenames.sort()
    for filename in filenames:
        print(f"{idx}/{total}")
        idx += 1
        # if idx > 10:
        #     break
        match = re.search('(\d+)-(\d+)\.txt', filename)
        pid = match.group(1)
        qid = match.group(2)
        pq, a, d = read_q_from_p(pid, qid, path)
        if d == None:
            continue
        if idx == 1:
            tmp_d = d[1]

        if prev_pid == -1:
            prev_pid = pid

        if prev_pid != pid:
            prev_d = tmp_d
            prev_pid = pid
            tmp_d = d[1]

        pq_e = get_embeddings(model, pq, device)
        a_e = get_embeddings(model, a, device)
        prev_d_e = get_embeddings(model, prev_d, device)
        d_e = []
        d_e.append(get_embeddings(model, d[0], device))
        d_e.append(get_embeddings(model, d[1], device))
        d_e.append(get_embeddings(model, d[2], device))

        pq_e_base = get_embeddings(model_base, pq, device)
        a_e_base = get_embeddings(model_base, a, device)
        prev_d_e_base = get_embeddings(model_base, prev_d, device)
        d_e_base = []
        d_e_base.append(get_embeddings(model_base, d[0], device))
        d_e_base.append(get_embeddings(model_base, d[1], device))
        d_e_base.append(get_embeddings(model_base, d[2], device))

        logging.info(f"\npid: {pid}, qid: {qid}")

        logging.info(f"\tanswer: {a}")
        sim_pq_d, sim_ans_d, sc_pq_ans, sc_ans_di, sc_avg = \
                calc_score(pq_e, a_e, a_e, max1, min1, max2, min2)
                # calc_score(pq_e, a_e, a_e, max1, min1, max2, min2)
        sim_pq_d_base, sim_ans_d_base, sc_pq_ans_base, sc_ans_di_base, sc_avg_base = \
                calc_score(pq_e_base, a_e_base, a_e_base, max1_base, min1_base, max2_base, min2_base)
                # calc_score(pq_e_base, a_e_base, a_e_base, max1_base, min1_base, max2_base, min2_base)
        logging.info(f"\tsc answer, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}, pq_d: {sim_pq_d}, ans_d: {sim_ans_d}")
        logging.info(f"\tsc_base answer, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}, pq_d: {sim_pq_d_base}, ans_d: {sim_ans_d_base}")

        sc_pq_ans, sc_ans_di = regulate_score(sc_pq_ans), regulate_score(sc_ans_di)
        sc_pq_ans_base, sc_ans_di_base = regulate_score(sc_pq_ans_base), regulate_score(sc_ans_di_base)
        sc_avg, sc_avg_base = (sc_pq_ans + sc_ans_di) / 2.0, (sc_pq_ans_base + sc_ans_di_base) / 2.0
        logging.info("\tAfter regulate:")
        logging.info(f"\tsc answer, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}")
        logging.info(f"\tsc_base answer, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}")

        logging.info(f"\tother d: {prev_d}")
        sim_pq_d, sim_ans_d, sc_pq_ans, sc_ans_di, sc_avg = \
                calc_score(pq_e, a_e, prev_d_e, max1, min1, max2, min2)
                # calc_score(pq_e, a_e, prev_d_e, max1, min1, max2, min2, a_e_2=a_e_base, target_e_2=prev_d_e_base)
        sim_pq_d_base, sim_ans_d_base, sc_pq_ans_base, sc_ans_di_base, sc_avg_base = \
                calc_score(pq_e_base, a_e_base, prev_d_e_base, max1_base, min1_base, max2_base, min2_base)
                # calc_score(pq_e_base, a_e_base, prev_d_e_base, max1_base, min1_base, max2_base, min2_base, a_e_2=a_e, target_e_2=prev_d_e)
        logging.info(f"\tsc other d, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}, pq_d: {sim_pq_d}, ans_d: {sim_ans_d}")
        logging.info(f"\tsc_base other d, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}, pq_d: {sim_pq_d_base}, ans_d: {sim_ans_d_base}")

        sc_pq_ans, sc_ans_di = regulate_score(sc_pq_ans), regulate_score(sc_ans_di)
        sc_pq_ans_base, sc_ans_di_base = regulate_score(sc_pq_ans_base), regulate_score(sc_ans_di_base)
        sc_avg, sc_avg_base = (sc_pq_ans + sc_ans_di) / 2.0, (sc_pq_ans_base + sc_ans_di_base) / 2.0
        logging.info("\tAfter regulate:")
        logging.info(f"\tsc other d, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}")
        logging.info(f"\tsc_base other d, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}")



        for j in range(3):
            logging.info(f"\t d[{j}]: {d[j]}")
            sim_pq_d, sim_ans_d, sc_pq_ans, sc_ans_di, sc_avg = \
                calc_score(pq_e, a_e, d_e[j], max1, min1, max2, min2)
                # calc_score(pq_e, a_e, d_e[j], max1, min1, max2, min2, a_e_2=a_e_base, target_e_2=d_e_base[j])
            # sim_pq_d = cosine_similarity(pq_e, d_e[j]).item()
            # sim_ans_d = cosine_similarity(a_e, d_e[j]).item()
            # sc_pq_ans = (sim_pq_d - min2) / (max2 - min2)
            # sc_ans_di = (max1 - sim_ans_d) / (max1 - min1)
            # sc_avg = (sc_pq_ans + sc_ans_di) / 2.0
            sim_pq_d_base, sim_ans_d_base, sc_pq_ans_base, sc_ans_di_base, sc_avg_base = \
                calc_score(pq_e_base, a_e_base, d_e_base[j], max1_base, min1_base, max2_base, min2_base)
                # calc_score(pq_e_base, a_e_base, d_e_base[j], max1_base, min1_base, max2_base, min2_base, a_e_2=a_e, target_e_2=d_e[j])
            # sim_pq_d_base = cosine_similarity(pq_e_base, d_e_base[j]).item()
            # sim_ans_d_base = cosine_similarity(a_e_base, d_e_base[j]).item()
            # sc_pq_ans_base = (sim_pq_d_base - min2_base) / (max2_base - min2_base)
            # sc_ans_di_base = (max1_base - sim_ans_d_base) / (max1_base - min1_base)
            # sc_avg_base = (sc_pq_ans_base + sc_ans_di_base) / 2.0
            logging.info(f"\tsc {j}, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}, pq_d: {sim_pq_d}, ans_d: {sim_ans_d}")
            logging.info(f"\tsc_base {j}, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}, pq_d: {sim_pq_d_base}, ans_d: {sim_ans_d_base}")
            sc_pq_ans, sc_ans_di = regulate_score(sc_pq_ans), regulate_score(sc_ans_di)
            sc_pq_ans_base, sc_ans_di_base = regulate_score(sc_pq_ans_base), regulate_score(sc_ans_di_base)
            sc_avg, sc_avg_base = (sc_pq_ans + sc_ans_di) / 2.0, (sc_pq_ans_base + sc_ans_di_base) / 2.0
            logging.info("\tAfter regulate:")
            logging.info(f"\tsc {j}, average score: {sc_avg}, pq ans score: {sc_pq_ans}, ans di score: {sc_ans_di}")
            logging.info(f"\tsc_base {j}, average score: {sc_avg_base}, pq ans score {sc_pq_ans_base}, ans di score: {sc_ans_di_base}")

            bleu_score_ans = bleu_score(d[j], a)
            bert_score_ans_p, bert_score_ans_r, bert_score_ans_f = score([a], [d[j]], lang="en", model_type="bert-base-uncased")
            logging.info(f"\t\tanswer bleu score, {bleu_score_ans}")
            logging.info(f"\t\tanswer bert score, f1: {bert_score_ans_f.item()}, precision: {bert_score_ans_p.item()}, recall: {bert_score_ans_r.item()}")

            bleu_score_other = bleu_score(d[j], prev_d)
            bert_score_other_p, bert_score_other_r, bert_score_other_f = score([prev_d], [d[j]], lang="en", model_type="bert-base-uncased")
            logging.info(f"\t\tother d bleu score, {bleu_score_other}")
            logging.info(f"\t\tother d bert score, f1: {bert_score_other_f.item()}, precision: {bert_score_other_p.item()}, recall: {bert_score_other_r.item()}")

            for k in range(3):
                bleu_score_k = bleu_score(d[j], d[k])
                # bleu_score_1 = bleu_score(d[j], d[1])
                # bleu_score_2 = bleu_score(d[j], d[2])

                bert_score_k_p, bert_score_k_r, bert_score_k_f = score([d[k]], [d[j]], lang="en", model_type="bert-base-uncased")
                # bert_score_1_p, bert_score_1_r, bert_score_1_f = score([d[1]], [d[j]], lang="en", model_type="bert-base-uncased")
                # bert_score_2_p, bert_score_2_r, bert_score_2_f = score([d[2]], [d[j]], lang="en", model_type="bert-base-uncased")

                # logging.info(f"d[{j}] {d[j]} as reference")
                logging.info(f"\t\tbleu score {k}, {bleu_score_k}")
                logging.info(f"\t\tbert score {k}, f1: {bert_score_k_f.item()}, precision: {bert_score_k_p.item()}, recall: {bert_score_k_r.item()}")
                # logging.info(f"\t\tbert score 1, precision: {bert_score_1_p.item()}, recall: {bert_score_1_r.item()}, f1: {bert_score_1_f.item()}")
                # logging.info(f"bert score 2, precision: {bert_score_2_p.item()}, recall: {bert_score_2_r.item()}, f1: {bert_score_2_f.item()}")

    #    # 输入文本
    # text1 = "This is an example sentence."
    # text2 = "Here is another example."

    # # 获取输入文本的词嵌入
    # embeddings1 = get_embeddings(text1)
    # embeddings2 = get_embeddings(text2)

    # # 计算余弦相似度
    # similarity = cosine_similarity(embeddings1, embeddings2)


# 输入文本
# reference_sentence = "This is an example sentence."
# candidate_sentence = "Here is another example."

# # 计算BLEU评分
# score = bleu_score(reference_sentence, candidate_sentence)

# print("BLEU score between the two sentences:", score)


# # 输入文本
# reference_sentence = "This is an example sentence."
# candidate_sentence = "Here is another example."

# # 计算bert-score
# P, R, F1 = score([candidate_sentence], [reference_sentence], lang="en", model_type="bert-base-uncased")

# print("BERT Score Precision:", P.item())
# print("BERT Score Recall:", R.item())
# print("BERT Score F1:", F1.item())

    # s = 'corr: {:10f}'.format(similarity.item())
    # logs_path = os.path.join(cur_dir, 'result-logs.txt')
    # with open(logs_path, 'a+') as f:
    #     s += '\n'
    #     f.write(s)
    # print("Cosine similarity between the two sentences:", similarity.item()) 