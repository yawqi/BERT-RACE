import random
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import json
import glob
import os
import spacy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# 加载预训练的GloVe词向量
glove_file = '.vector_cache/glove.840B.300d.txt'
nlp = spacy.load('en_core_web_sm')
# word2vec_output_file = '.vector_cache/glove.840B.300d.word2vec.txt'

# # 将GloVe文件转换为Word2Vec格式
# if not os.path.exists(word2vec_output_file):
#     print("converting")
#     glove2word2vec(glove_file, word2vec_output_file)
    # print("convert over")
print("loading")
word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
print("loading over")

# 定义一个函数来查找每个单词的最接近的近义词列表
def find_similar_words(word, n=3):
    if word not in word_vectors:
        print("{} missing".format(word))
        return []
    similar_words = word_vectors.most_similar(word, topn=n)
    if len(similar_words) == 0:
        print("{} no similar words".format(word))
        return []
    # print("{} : {}".format(word, similar_words))
    return [w[0] for w in similar_words]

# 定义一个函数来替换句子中一半的单词
def replace_words(sentence):
    words = sentence.split()
    n = len(words)
    replace_indices = random.sample(range(n), max(1, n//2))
    for i in replace_indices:
        word = words[i]
        # 对单词进行词形还原和归一化处理
        token = nlp(word)[0]
        lemma = token.lemma_.lower()
        similar_words = find_similar_words(lemma)
        if similar_words:
            words[i] = random.choice(similar_words)
    return ' '.join(words)

def shuffle_words(answer):
    words = answer.split()
    random.shuffle(words)
    return " ".join(words)

def read_data_from_path(path, type):
    examples = []
    filenames = glob.glob(path + '/*')
    for filename in filenames:
        print("Open file %s" % filename)
        with open(filename, 'r', encoding='utf-8') as fpr:
            # pid = int(match.group(1))
            data_raw = json.load(fpr)
            if data_raw['type'] == type:
                examples.append([data_raw['s1'], data_raw['s2']])
    return examples

def dump_data(cn, path, type, idx = 0):
    count = idx
    if not os.path.exists(path):
        os.makedirs(path)
    for i, [s1, s2] in enumerate(cn):
        data = {}
        data['s1'] = s1
        data['s2'] = s2
        data['type'] = int(type)
        filename = os.path.join(path, str(count) + '.txt')
        count += 1
        with open(filename, 'w') as f:
            json.dump(data, f)
    return count

def process_replace_words_type_0(c2_type_0, i):
    c2_type_0[i][0] = replace_words(c2_type_0[i][0])
    if i % 100 == 0:
        print("{} done, {} to do".format(i, len(c2_type_0) - i))
    return c2_type_0[i]

def process_replace_words_type_2(c2_type_2, i):
    c2_type_2[i][1] = replace_words(c2_type_2[i][1])
    if i % 100 == 0:
        print("{} done, {} to do".format(i, len(c2_type_2) - i))
    return c2_type_2[i]


def generate_C2(c1, c4, c5, num_threads=32):
    type_0_count = len(c1) // 2
    type_1_count = len(c1) // 4
    type_2_count = len(c1) - type_0_count - type_1_count
    c2_type_0 = random.sample(c1, type_0_count)
    c2_type_1 = []
    c2_type_2 = random.sample(c1, type_2_count)
    # for i in range(0, type_0_count):
    #     if i % 100 == 0:
    #         print("{} done, {} to do".format(i, type_0_count - i))
    #     c2_type_0[i][0] = replace_words(c2_type_0[i][0])
    # c2_type_1.extend(random.sample(c4, type_1_count // 2))
    # c2_type_1.extend(random.sample(c5, type_1_count - type_1_count // 2))
    # for i in range(0, type_2_count):
    #     if i % 100 == 0:
    #         print("{} done, {} to do".format(i, type_2_count - i))
    #     c2_type_2[i][1] = replace_words(c2_type_2[i][1])
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        process_replace_words_type_0_partial = partial(process_replace_words_type_0, c2_type_0)
        c2_type_0 = list(executor.map(process_replace_words_type_0_partial, range(type_0_count)))

        process_replace_words_type_2_partial = partial(process_replace_words_type_2, c2_type_2)
        c2_type_2 = list(executor.map(process_replace_words_type_2_partial, range(type_2_count)))

    c2_type_1.extend(random.sample(c4, type_1_count // 2))
    c2_type_1.extend(random.sample(c5, type_1_count - type_1_count // 2))
    return c2_type_0, c2_type_1, c2_type_2

# 示例用法
race_sr_dir = './RACE-SR-NEW'
test_dir = os.path.join(race_sr_dir, 'test')
dev_dir = os.path.join(race_sr_dir, 'dev')
train_dir = os.path.join(race_sr_dir, 'train')

out_dir = './TASK-2-DATA'
test_out_dir = os.path.join(out_dir, 'test/C2')
dev_out_dir = os.path.join(out_dir, 'dev/C2')
train_out_dir = os.path.join(out_dir, 'train/C2')

input_dir = test_dir
output_dir = test_out_dir

c1 = read_data_from_path(os.path.join(input_dir, 'C3'), 0)
c4 = read_data_from_path(os.path.join(input_dir, 'C4'), 2)
c5 = read_data_from_path(os.path.join(input_dir, 'C5'), 2)
c2_type_0, c2_type_1, c2_type_2 = generate_C2(c1, c4, c5)
count = dump_data(c2_type_0, output_dir, 0)
count = dump_data(c2_type_1, output_dir, 1, count)
count = dump_data(c2_type_2, output_dir, 2, count)