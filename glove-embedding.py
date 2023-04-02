from torchtext.vocab import GloVe
import spacy
import torch
import glob 
import json
import os
import nltk
import numpy as np

glove = GloVe(name='840B', dim=300)
nlp = spacy.load('en_core_web_sm')
# nltk.download("punkt")
data_dir = "./RACE"
train_dir = os.path.join(data_dir, 'train')
dev_dir = os.path.join(data_dir, 'dev')
test_dir = os.path.join(data_dir, 'test')

new_dir = "./NEW-RACE200-train"
new_dev_dir = os.path.join(new_dir, 'dev')
new_train_dir = os.path.join(new_dir, 'train')
new_test_dir = os.path.join(new_dir, 'test')

def tokenize(text):
    tokens = nlp(text)
    ret = []
    for t in tokens:
        ret.append(t.text)
    return ret

def embed_sentence(sentence):
    vectors = []
    tokens = tokenize(sentence)
    for word in tokens:
        # print("embed word: %s" % word)
        if word in glove.stoi:
            # print("embeding")
            vectors.append(glove[word])
        # print("embed over")
    if not vectors:
        vectors = [torch.zeros(glove.dim)]
    return torch.mean(torch.stack(vectors), axis = 0)

# import torch
# from torch.utils.data import Dataset

# class RACE_Dataset(Dataset):
#     def __init__(self, data):
#         self.data = data
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         article = self.data[idx]['article']
#         options = self.data[idx]['options']
#         labels = self.data[idx]['labels']
        
#         article_vecs = embed_sentence(article)
#         options_vecs = [embed_sentence(option) for option in options]
        
#         return {'article_vecs': article_vecs, 'options_vecs': options_vecs, 'labels': labels}

def read_race_examples(paths):
    # examples = []
    articles = []
    questions = []
    answers = []
    dis = []
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                articles.append(nltk.sent_tokenize(article))
                article_q = []
                article_a = []
                article_di = []
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    article_q.append(question)
                    article_a.append(options[truth])
                    article_di.append(options)
                    # examples.append(
                    #     RaceExample(
                    #         race_id = filename+'-'+str(i),
                    #         context_sentence = article,
                    #         start_ending = question,

                    #         ending_0 = options[0],
                    #         ending_1 = options[1],
                    #         ending_2 = options[2],
                    #         ending_3 = options[3],
                    #         label = truth))
                    
                questions.append(article_q)
                answers.append(article_a)
                dis.append(article_di)
    return articles, questions, answers, dis

def vectorize(sentences):
    v = []
    # print("vectorizing: %s" % sentences)
    for sentence in sentences:
        v.append(embed_sentence(sentence))
    return v

def find_top_k_similarities(vectors, target_vector, k):
    k = min(k, len(vectors))
    # 计算目标向量与所有向量的余弦相似度
    similarities = [np.dot(target_vector, v.t()) / (np.linalg.norm(target_vector) * np.linalg.norm(v)) for v in vectors]
    # 对余弦相似度排序，找出前k个最大值对应的索引
    top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    # 返回前k个最大余弦相似度对应的向量和相似度值
    return [(vectors[i], similarities[i]) for i in top_k_indices], top_k_indices

def main():
    articles, questions, answers, dis = read_race_examples(['./RACE200-train'])
    v_articles = [vectorize(v) for v in articles]
    v_qas = []

    for i in range(len(questions)):
        # print("making sentence iter: %d" % i)
        qa = []
        for j in range(len(questions[i])):
            qa.append(embed_sentence(questions[i][j] + ' ' + answers[i][j]))
        v_qas.append(qa)

    PQs = []
    for pid, qas in enumerate(v_qas):
        # print("computing iter: %d" % idx)
        PQ = []
        data = {}
        for qid, qa in enumerate(qas):
            pq = ""
            answer = answers[pid][qid]
            top_k, top_k_idx = find_top_k_similarities(v_articles[pid], qa, 3)
            for i in top_k_idx:
                pq += articles[pid][i] + " "
            pq += questions[pid][qid]
            PQ.append(pq)
            D = []
            for d in dis[pid][qid]:
                if d != answer:
                    D.append(d)
            print("pid %d qid %d: PQ: %s A: %s D: %s" % (pid, qid, pq, answer, D))
            filename = str(pid) + '-' + str(qid) + '.txt'
            data['PQ'] = pq
            data['A'] = answer
            data['D'] = D
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            with open(os.path.join(new_dir, filename), 'w') as f:
                json.dump(data, f) 
        # PQs.append(PQ)
    # print(PQs)

if __name__ == "__main__":
    main()