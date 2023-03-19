

import random
import numpy as np
from gensim.models import KeyedVectors

# 加载预训练的GloVe词向量
glove_file = '.vector_cache/glove.840B.300d.txt'
word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False)

# 定义一个函数来查找每个单词的最接近的近义词列表
def find_similar_words(word, n=3):
    similar_words = word_vectors.most_similar(word, topn=n)
    return [w[0] for w in similar_words]

# 定义一个函数来替换句子中一半的单词
def replace_words(sentence):
    words = sentence.split()
    n = len(words)
    replace_indices = random.sample(range(n), n//2)
    for i in replace_indices:
        word = words[i]
        similar_words = find_similar_words(word)
        if similar_words:
            words[i] = random.choice(similar_words)
    return ' '.join(words)

def shuffle_words(answer):
    words = answer.split()
    random.shuffle(words)
    return " ".join(words)

# 示例用法
sentence = "The cat sat on the mat."
new_sentence = replace_words(sentence)
print(new_sentence)
