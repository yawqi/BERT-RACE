import re
import glob
import json

word_dict = {}
def read_data_from_path(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path + '/*')
        for filename in filenames:
            print("Open file %s" % filename)
            with open(filename, 'r', encoding='utf-8') as fpr:
                # pid = int(match.group(1))
                data_raw = json.load(fpr)
                examples.append({'PQ': data_raw['PQ'], 'A': data_raw['A'], 'D': data_raw['D']})
    return examples

def split_count(s):
    lst = re.split(r'\W+', s) # 使用正则表达式\W+进行分词
    global word_dict
    for word in lst: # 遍历分词后的列表
        word_dict[word] = word_dict.get(word, 0) + 1 # 统计单词出现的次数
    return len(lst)

# s = 'How are you! How is it going? How about some coffee?'
# result = split_count(s)
# print(result)

def main():
    src_dir_dev = 'NEW-RACE-DEV'
    src_dir_test = 'NEW-RACE-TEST'
    src_dir_train = 'NEW-RACE-TRAIN'
    l = read_data_from_path([src_dir_dev, src_dir_test, src_dir_train])
    # l = read_data_from_path([src_dir_dev,])
    pq_len, a_len, d_len = 0.0, 0.0, 0.0
    pq_count, a_count, d_count = 0, 0, 0
    for d in l:
        pq_count += 1
        a_count += 1
        d_count += len(d['D'])
        pq_len += float(split_count(d['PQ']))
        a_len += float(split_count(d['A']))
        for di in d['D']:
            d_count += 1
            d_len += float(split_count(di))
    pq_avg = pq_len / float(pq_count)
    a_avg = a_len / float(a_count)
    d_avg = d_len / float(d_count)
    print(f'pq: {pq_avg}, a: {a_avg}, d: {d_avg}, vocab: {len(word_dict)}')
if __name__ == '__main__':
    main()