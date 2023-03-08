import numpy as np
import os

import time
from utils import *
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from collections import defaultdict as ddict
from sklearn.metrics.pairwise import cosine_similarity as cos
import torchtext.vocab as vocab
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-dataset', default='./apr', help='path to dataset folder')
parser.add_argument('-output', default='./apr/results',
                    help='file name for output')
args = parser.parse_args()

glove = vocab.pretrained_aliases["glove.840B.300d"](cache='./cache')

connect = []
disconnect = []
eid2name, keywords, eid2idx = load_vocab(os.path.join(args.dataset, 'entity2id.txt'))

for i in keywords:
    entity_name = eid2name[i]
    entity_name = entity_name.split()
    entity_name = ''.join(entity_name[:])
    if entity_name in glove.stoi:
        connect.append(i)
    else:
        disconnect.append(i)


def pre_expan(seed, threshold):
    seed_1 = seed[:]
    query_sets = []
    for i in seed_1:
        entity_name = eid2name[i]
        entity_name = entity_name.split()
        joint = ''.join(entity_name[:])
        if joint not in glove.stoi:
            emb = [glove.vectors[glove.stoi[j]].numpy() for j in entity_name]
            cc = np.mean(emb, axis=0)
            query_sets.append(cc)
        else:
            cc = glove.vectors[glove.stoi[joint]]
            query_sets.append(cc.numpy())
    dd = []
    qq = []
    for i in keywords:
        entity_name = eid2name[i]
        entity_name = entity_name.split()
        entity_name = ''.join(entity_name[:])
        if entity_name in glove.stoi:
            cc = glove.vectors[glove.stoi[entity_name]]
            aaa = cc.numpy()
            qq.append(i)
            dd.append(aaa)
    scores = cos(query_sets, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)
    this_global_score = []
    for i in score_ranking:
        if scores[i] > threshold:
            this_global_score.append(i)
    for i in this_global_score:
        count = 0
        for j in qq:
            if i == count:
                if j not in seed:
                    seed_1.append(j)
            count += 1
    return seed_1


def candidate(query_sets, threshold):
    result = []

    query_sets_1 = []
    flag = 0
    for i in query_sets:
        entity_name = eid2name[i]
        entity_name = entity_name.split()
        
        for m in entity_name:
            if m not in glove.stoi:
                flag = 0
                break
            flag = 1
        if flag: 
            joint = ''.join(entity_name[:])
            if joint not in glove.stoi:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in entity_name]
                cc = np.mean(emb, axis=0)
                query_sets_1.append(cc)
            else:
                cc = glove.vectors[glove.stoi[joint]]
                query_sets_1.append(cc.numpy())
    dd = []
    qq = []
    flag = 0
    for i in disconnect:
        if i not in query_sets:
            entity_name = eid2name[i]
            entity_name = entity_name.split()
            for j in entity_name:
                if j not in glove.stoi:
                    flag = 0
                    break
                flag = 1
            if flag:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in entity_name]
                qq.append(i)
                aaa = np.mean(emb, axis=0)
                dd.append(aaa)
    scores = cos(query_sets_1, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)

    this_keywords1 = [qq[i]
                      for i in score_ranking if scores[i] > threshold]
    for mm in this_keywords1:
        if mm not in result:
            result.append(mm)

    dd = []
    qq = []
    for i in connect:
        if i not in query_sets:
            entity_name = eid2name[i]
            entity_name = entity_name.split()
            for j in entity_name:
                if j not in glove.stoi:
                    flag = 0
                    break
                flag = 1
            if flag:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in entity_name]
                qq.append(i)
                aaa = np.mean(emb, axis=0)
                dd.append(aaa)

    scores = cos(query_sets_1, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)

    this_keywords1 = [qq[i]
                      for i in score_ranking if scores[i] > threshold]

    for mm in this_keywords1:
        if mm not in result:
            result.append(mm)

    dd = []
    qq = []
    for i in connect:
        if i not in query_sets:
            entity_name = eid2name[i]
            entity_name = entity_name.split()
            entity_name = ''.join(entity_name[:])
            if entity_name in glove.stoi:
                cc = glove.vectors[glove.stoi[entity_name]]
                aaa = cc.numpy()
                qq.append(i)
                dd.append(aaa)
    scores = cos(query_sets_1, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)

    this_keywords = [qq[i]
                     for i in score_ranking if scores[i] > threshold]

    return result+this_keywords


class Expan(object):
    def __init__(self, device, model_name='bert-base-uncased', dim=768):
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=False)

        self.maskedLM = BertForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True)
        self.maskedLM.to(device)
        self.maskedLM.eval()

        self.eid2name, self.keywords, self.eid2idx = load_vocab(
            os.path.join(args.dataset, 'entity2id.txt'))
        self.entity_pos = pickle.load(
            open(os.path.join(args.dataset, 'entity_pos.pkl'), 'rb'))

        self.pretrained_emb = np.memmap(os.path.join(args.dataset, 'pretrained_emb.npy'),
                                        dtype='float32', mode='r', shape=(self.entity_pos[-1], dim))

        self.means = np.array([np.mean(emb, axis=0)
                               for emb in self.get_emb_iter()])

    def rand_idx(self, l):
        for _ in range(10000):
            for i in np.random.permutation(l):
                yield i

    def get_emb_iter(self):
        for i in range(len(self.keywords)):
            yield self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i+1]]

    def expand(self, query_set, expan_glove, candidate_pool, target_size, gt=None):
        print('start expanding: ' +
              str([self.eid2name[eid] for eid in query_set]))
        start_time = time.time()
        expanded_set = expan_glove[:]
        while len(expanded_set) < target_size:
            print(
                f'num of expanded entities: {len(expanded_set)}, time: {int((time.time() - start_time)/60)} min {int(time.time() - start_time)%60} sec')
            if gt is not None:
                print(
                    f'map10: {apk(gt, expanded_set, 10)}, map20: {apk(gt, expanded_set, 20)}, map50: {apk(gt, expanded_set, 50)}')

            new_entities = self.expansion(
                query_set + expanded_set, candidate_pool)
            expanded_set.extend(new_entities)
        return expanded_set

    def expansion(self, current_set, candidate_pool):
        idx_generator = self.rand_idx(len(current_set))
        cos_scores = cos(self.means[[self.eid2idx[eid]
                                     for eid in current_set]], self.means[[self.eid2idx[eid] for eid in candidate_pool]])
        eid2mrr = ddict(float)
        indices = []
        for n in idx_generator:
            if n not in indices:
                indices.append(n)
                if len(indices) == 3:
                    break
        # mean_score = np.mean(cos_scores[indices], axis=0)
        mean_score = np.mean(cos_scores[:], axis=0)
        mean_score_ranking = np.argsort(-mean_score)
        this_keywords = [candidate_pool[i]
                         for i in mean_score_ranking[:500]]
        mean_score = [mean_score[i]
                      for i in mean_score_ranking[:500]]
        scores = np.array(mean_score)
        r = 0.
        for i in np.argsort(-scores):
            eid = this_keywords[i]
            if eid not in set(current_set):
                r += 1
                eid2mrr[eid] += 1 / r
            if r >= 20:
                break
        eid_rank = sorted(eid2mrr, key=lambda x: eid2mrr[x], reverse=True)

        return eid_rank[:1]


if __name__ == '__main__':
    thres_l, thres_h = 0.25, 0.65

    expan_1 = Expan(torch.device("cuda:0"))

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for file in os.listdir(os.path.join(args.dataset, 'query')):
        query_sets = []
        with open(os.path.join(args.dataset, 'query', file), encoding='utf-8') as f:
            for line in f:
                if line == 'EXIT\n':
                    break
                temp = line.strip().split(' ')
                query_sets.append([int(eid) for eid in temp])
        gt = set()
        with open(os.path.join(args.dataset, 'gt', file), encoding='utf-8') as f:
            for line in f:
                temp = line.strip().split('\t')
                eid = int(temp[0])
                if int(temp[2]) >= 1:
                    gt.add(eid)
        score = 0
        for i in range(len(query_sets)):
            expan_glove = pre_expan(query_sets[i], thres_h)
            candidate_pool = candidate(expan_glove, thres_l)
            expanded = expan_1.expand(
                query_sets[i], expan_glove, candidate_pool, 50,  gt)
            with open(os.path.join(args.output, f'{i}_{file}'), 'w') as f:
                print(apk(gt, expanded, 10), file=f)
                print(apk(gt, expanded, 20), file=f)
                print(apk(gt, expanded, 50), file=f)
                print(apk(gt, expanded, 50))
                score += apk(gt, expanded, 50)
                print('', file=f)
                for eid in expanded:
                    print(f'{eid}\t{expan_1.eid2name[eid]}', file=f)
        print(score/5)
