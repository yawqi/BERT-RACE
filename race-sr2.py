import re
import glob
import json
import os
import random

data_dir = './NEW-RACE'
train_data_dir = data_dir + '-TRAIN'
test_data_dir = data_dir + '-TEST'
dev_data_dir = data_dir + '-DEV'
race_sr_dir = './RACE-SR-C4'
test_dir = os.path.join(race_sr_dir, 'test')
dev_dir = os.path.join(race_sr_dir, 'dev')
train_dir = os.path.join(race_sr_dir, 'train')

input_dir = train_data_dir
output_dir = train_dir

c1_dir = os.path.join(output_dir, "C1")
c2_dir = os.path.join(output_dir, "C2")
c3_dir = os.path.join(output_dir, "C3")
c4_dir = os.path.join(output_dir, "C4")
# c5_dir = os.path.join(output_dir, "C5")
# c6_dir = os.path.join(cur_dir, "C6")
c1 = []
c2 = []
c3 = []
c4 = []
# c5 = []
# c6 = []
# pqa_or_pqd = 0
c3_opt = 0
c4_opt = 0

def generate_data(pid1, pid2, qid1, qid2, all_pqs, all_as, all_ds):
    if len(all_ds[pid2][qid2]) == 0:
        return
    d_idx = random.randint(0, len(all_ds[pid2][qid2]) - 1)
    pq = all_pqs[pid1][qid1]
    a = all_as[pid2][qid2]
    d = all_ds[pid2][qid2][d_idx]
    global c3_opt, c4_opt
    if pid1 == pid2:
        if qid1 == qid2:
            # 同一P、Q
            c1.append([pq, a, '0'])
            c2.append([pq, d, '0'])
            # c3.append([a, d, '0'])
        else:
            # 同一P、不同Q
            if c3_opt == 0:
                c3.append([pq, a, '0'])
            elif c3_opt == 1:
                c3.append([pq, d, '1'])
            # else:
            #     c4.append([a, d, '2'])
            c3_opt = (c3_opt + 1) % 2
    else:
        # 不同P、不同Q
        if c4_opt == 0:
            c4.append([pq, a, '0'])
        elif c4_opt == 1:
            c4.append([pq, d, '1'])
        # else:
        #     c5.append([a, d, '2'])
        c4_opt = (c4_opt + 1) % 2

def read_race_examples(paths):
    all_pqs = []
    all_as = []
    all_ds = []
    article_pqs = []
    article_as = []
    article_ds = []
    prev_pid = 0
    pid = 0
    qid = 0
    pcount = 0
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        filenames = sorted(filenames)
        for filename in filenames:
            print("Opening %s" % filename)
            with open(filename, 'r', encoding='utf-8') as fpr:
                match = re.search(r'(\d+)-(\d+)', filename)
                pid = int(match.group(1))
                if pid != prev_pid:
                    all_pqs.append(article_pqs)
                    all_as.append(article_as)
                    all_ds.append(article_ds)
                    article_pqs = []
                    article_as = []
                    article_ds = []
                    for q in range(qid + 1):
                        generate_data(pcount, pcount, q, q, all_pqs, all_as, all_ds)
                        if qid == 0:
                            break
                        q2 = q
                        while q2 == q:
                            q2 = random.randint(0, qid)
                        generate_data(pcount, pcount, q, q2, all_pqs, all_as, all_ds)
                    if pcount % 2 == 1:
                        for q1 in range(qid + 1):
                            q2 = random.randint(0, len(all_pqs[pcount - 1]) - 1)
                            generate_data(pcount, pcount - 1, q1, q2, all_pqs, all_as, all_ds)
                        for q1 in range(len(all_pqs[pcount - 1])):
                            q2 = random.randint(0, qid)
                            generate_data(pcount - 1, pcount, q1, q2, all_pqs, all_as, all_ds)
                    prev_pid = pid
                    pcount += 1
                qid = int(match.group(2))
                # print("reading pid: %d qid: %d" % (pid, qid))
                data_raw = json.load(fpr)
                PQ = data_raw['PQ']
                A = data_raw['A']
                D = data_raw['D']

                # print("PQ: %s A: %s D: %s" % (PQ, A, D))
                article_pqs.append(PQ)
                article_as.append(A)
                article_ds.append(D)

def dump_data(cn, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, [s1, s2, type] in enumerate(cn):
        data = {}
        data['s1'] = s1
        data['s2'] = s2
        data['type'] = int(type)
        filename = os.path.join(path, str(i) + '.txt')
        with open(filename, 'w') as f:
            json.dump(data, f)
    
def main():
    read_race_examples([input_dir,])
    dump_data(c1, c1_dir)
    dump_data(c2, c2_dir)
    dump_data(c3, c3_dir)
    dump_data(c4, c4_dir)
    # dump_data(c5, c5_dir)
    # dump_data(c6, c6_dir)
if __name__ == '__main__':
    main()