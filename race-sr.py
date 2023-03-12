import re
import glob
import json
import os
import random

data_dir = './NEW-RACE'
train_data_dir = data_dir + '-TRAIN'
test_data_dir = data_dir + '-TEST'
dev_data_dir = data_dir + '-DEV'
race_sr_dir = './RACE-SR'
test_dir = os.path.join(race_sr_dir, 'test')
dev_dir = os.path.join(race_sr_dir, 'dev')
train_dir = os.path.join(race_sr_dir, 'train')
cur_dir = train_dir
c1_dir = os.path.join(cur_dir, "C1")
c2_dir = os.path.join(cur_dir, "C2")
c3_dir = os.path.join(cur_dir, "C3")
c4_dir = os.path.join(cur_dir, "C4")
c5_dir = os.path.join(cur_dir, "C5")
c6_dir = os.path.join(cur_dir, "C6")
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
pqa_or_pqd = 0
c6_opt = 0

def generate_data(pid1, pid2, qid1, qid2, all_pqs, all_as, all_ds):
    if len(all_ds[pid2][qid2]) == 0:
        return
    d_idx = random.randint(0, len(all_ds[pid2][qid2]) - 1)
    pq = all_pqs[pid1][qid1]
    a = all_as[pid2][qid2]
    d = all_ds[pid2][qid2][d_idx]
    global pqa_or_pqd, c6_opt
    if pid1 == pid2:
        if qid1 == qid2:
            # 同一P、Q
            c1.append([pq, a])
            c2.append([pq, d])
            c3.append([a, d])
        else:
            # 同一P、不同Q
            if pqa_or_pqd % 2 == 0:
                c4.append([pq, a])
            else:
                c4.append([pq, d])
            c5.append([a, d])
            pqa_or_pqd += 1
    else:
        # 不同P、不同Q
        if c6_opt == 0:
            c6.append([pq, a])
        elif c6_opt == 1:
            c6.append([pq, d])
        else:
            c6.append([a, d])
        c6_opt = (c6_opt + 1) % 3

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
                    # print("pqs: %s", article_pqs)
                    # print("as: %s", article_as)
                    # print("ds: %s", article_ds)
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
                # print("test")
        # print("End path: %s", path)
    # print("End read")
        # last round
        # all_pqs.append(artiqle_pqs)
        # all_as.append(article_as)
        # all_ds.append(article_ds)
        # artiqle_pqs = []
        # article_as = []
        # article_ds = []
        # for q in range(qid):
        #     generate_data(prev_pid, prev_pid, q, q, all_pqs, all_as, all_ds)
        #     q2 = q
        #     while q2 == q:
        #         q2 = random.randint(0, qid - 1)
        #     generate_data(prev_pid, prev_pid, q, q2, all_pqs, all_as, all_ds)
        # if prev_pid % 2 == 1:
        #     for q1 in range(qid + 1):
        #         q2 = random.randint(0, len(all_pqs[prev_pid - 1] - 1))
        #         generate_data(prev_pid, prev_pid - 1, q1, q2)
        #     for q1 in range(len(all_pqs[prev_pid - 1])):
        #         q2 = random.randint(0, qid)
        #         generate_data(prev_pid - 1, prev_pid, q1, q2)
def dump_data(cn, path):
    for i, [s1, s2] in enumerate(cn):
        data = {}
        data['s1'] = s1
        data['s2'] = s2
        filename = os.path.join(path, str(i) + '.txt')
        with open(filename, 'w') as f:
            json.dump(data, f)
    
def main():
    read_race_examples([train_data_dir,])
    dump_data(c1, c1_dir)
    dump_data(c2, c2_dir)
    dump_data(c3, c3_dir)
    dump_data(c4, c4_dir)
    dump_data(c5, c5_dir)
    dump_data(c6, c6_dir)
if __name__ == '__main__':
    main()