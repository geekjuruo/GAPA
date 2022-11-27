import numpy as np
import os
import inflect
import nltk
import copy
import queue
import time
from utils import *
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from collections import defaultdict as ddict
from sklearn.metrics.pairwise import cosine_similarity as cos
import torchtext.vocab as vocab
import torch
import argparse
from transformers import RobertaTokenizer, RobertaModel

glove = vocab.pretrained_aliases["glove.840B.300d"](cache='./cache')


def load_vocab(filename):
    eid2name = {}
    keywords = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            keywords.append(eid)
    eid2idx = {w: i for i, w in enumerate(keywords)}
    return eid2name, keywords, eid2idx


connect = []
disconnect = []
eid2name, keywords, eid2idx = load_vocab('./apr/entity2id.txt')


for i in keywords:
        hh = eid2name[i]
        hh = hh.split()
        hh = ''.join(hh[:])
        if hh in glove.stoi:
            connect.append(i)
        else:
            disconnect.append(i)


def pre_expan(a, kkkkk, lll):
    num_qianduoshaoge = 8
    num_qianduoshaoge_1 = 10
    num_qianduoshaoge_2 = 26
    a_1 = a[:]
    query_sets = []
    for i in a:
        hh = eid2name[i]
        hh = hh.split()
        lianjie = ''.join(hh[:])        
        if lianjie not in glove.stoi:
            if lll == 2:
                return a
            emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
            cc = np.mean(emb, axis=0)
            query_sets.append(cc)
        else:
            cc = glove.vectors[glove.stoi[lianjie]]
            query_sets.append(cc.numpy())
    dd = []
    qq = []
    for i in keywords:
        hh = eid2name[i]        
        hh = hh.split()        
        hh = ''.join(hh[:])        
        if hh in glove.stoi:            
            cc = glove.vectors[glove.stoi[hh]]
            aaa = cc.numpy()
            qq.append(i)
            dd.append(aaa)
    scores = cos(query_sets, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)
    yy = [scores[i] for i in score_ranking[:num_qianduoshaoge+len(a)]]
    this_global_score = []
    for i in score_ranking[:num_qianduoshaoge+len(a)]:
        if scores[i] > 0:
            this_global_score.append(i)

    for i in this_global_score:
        count = 0
        for j in qq:
            if i == count:
                if j not in a:
                    a_1.append(j)
            count += 1

    query_sets = []
    for i in a_1:
        hh = eid2name[i]
        hh = hh.split()
        lianjie = ''.join(hh[:])
        if lianjie not in glove.stoi:
            emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
            cc = np.mean(emb, axis=0)
            query_sets.append(cc)
        
        else:
            cc = glove.vectors[glove.stoi[lianjie]]
            query_sets.append(cc.numpy())
    dd = []
    qq = []
    for i in keywords:
        hh = eid2name[i]
        hh = hh.split()
        hh = ''.join(hh[:])
        if hh in glove.stoi:
            cc = glove.vectors[glove.stoi[hh]]
            aaa = cc.numpy()
            qq.append(i)
            dd.append(aaa)
    scores = cos(query_sets, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)
    yy = [scores[i] for i in score_ranking[:num_qianduoshaoge_1+len(a)]]
    # print(yy)
    this_global_score = []
    for i in score_ranking[:num_qianduoshaoge_1+len(a)]:
        if scores[i] > 0:
            this_global_score.append(i)

    for i in this_global_score:
        count = 0
        for j in qq:
            if i == count:
                if j not in a_1:
                    a_1.append(j)
            count += 1

    query_sets = []
    for i in a_1:
        hh = eid2name[i]
        hh = hh.split()
        lianjie = ''.join(hh[:])
        if lianjie not in glove.stoi:
            emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
            cc = np.mean(emb, axis=0)
            query_sets.append(cc)        
        else:
            cc = glove.vectors[glove.stoi[lianjie]]
            query_sets.append(cc.numpy())
    dd = []
    qq = []
    for i in keywords:
        hh = eid2name[i]
        hh = hh.split()
        hh = ''.join(hh[:])
        if hh in glove.stoi:
            cc = glove.vectors[glove.stoi[hh]]
            aaa = cc.numpy()
            qq.append(i)
            dd.append(aaa)
    scores = cos(query_sets, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)
    yy = [scores[i] for i in score_ranking[:num_qianduoshaoge_2+len(a)]]
    this_global_score = []
    for i in score_ranking[:num_qianduoshaoge_2+len(a)]:
        if scores[i] > kkkkk:
            # print(scores[i])
            this_global_score.append(i)
    a_1 = a[:]
    flag_1 = 0
    for i in this_global_score:
        count = 0
        for j in qq:
            if i == count:
                if j not in a:
                    flag_1 = 1
                    a_1.append(j)
            count += 1

    if flag_1 == 0:
        return a
    else:
        return a_1


def candidate(query_sets):
    num1 = 0
    flag = 1
    for i in query_sets:
        if i in connect:
            num1 += 1
    if num1 == 3:
        num_qianduoshaoge = 250
        num_qianduoshaoge_1 = 750
        num_qianduoshaoge_2 = 20  
    else:
        num_qianduoshaoge = 350  
        num_qianduoshaoge_1 = 2000
        num_qianduoshaoge_2 = 20 
    result = []

    if len(query_sets) > 15:
        num_qianduoshaoge = 450  
        num_qianduoshaoge_1 = 3  
        num_qianduoshaoge_2 = 200  
        flag = 1

    if flag == 0:
        entity1 = []
        entity2 = []
        for i in query_sets:
            entity1.append(eid2name[i])
        for i in disconnect:
            if i not in query_sets:
                entity2.append(eid2name[i])
        embeddings1 = model.encode(entity1, convert_to_tensor=True)
        embeddings2 = model.encode(entity2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        mean_score = np.mean(cosine_scores.cpu().numpy()[:], axis=0)
        mean_score_ranking = np.argsort(-mean_score)
        this_keywords = [disconnect[i] for i in mean_score_ranking[:1000]]
        mean_score = [mean_score[i]
                      for i in mean_score_ranking[:1000]]  

        for n in this_keywords:
            result.append(n)

    query_sets_1 = []
    flag = 0
    for i in query_sets:
        hh = eid2name[i]
        hh = hh.split()
        
        for m in hh:
            if m not in glove.stoi:
                flag = 0
                break
            flag = 1
        if flag:
            lianjie = ''.join(hh[:])
            if lianjie not in glove.stoi:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
                cc = np.mean(emb, axis=0)
                query_sets_1.append(cc)
            else:
                cc = glove.vectors[glove.stoi[lianjie]]
                query_sets_1.append(cc.numpy())
    dd = []
    qq = []
    flag = 0
    for i in disconnect:
        if i not in query_sets:
            hh = eid2name[i]
            hh = hh.split()
            for j in hh:
                if j not in glove.stoi:
                    flag = 0
                    break
                flag = 1
            if flag:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
                qq.append(i)
                aaa = np.mean(emb, axis=0)
                dd.append(aaa)
    scores = cos(query_sets_1, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)

    this_score_ranking = [i
                          for i in score_ranking[:num_qianduoshaoge_1+len(query_sets)]]
    yy = [scores[i]
          for i in score_ranking[:num_qianduoshaoge_1+len(query_sets)]]
    this_keywords1 = [qq[i]
                      for i in score_ranking[:num_qianduoshaoge_1+len(query_sets)]]
    for mm in this_keywords1:
        if mm not in result:
            result.append(mm)

    query_sets_1 = []
    flag = 0
    for i in query_sets:
        hh = eid2name[i]
        hh = hh.split()
        
        for m in hh:
            if m not in glove.stoi:
                flag = 0
                break
            flag = 1
        if flag:
            lianjie = ''.join(hh[:])
            if lianjie not in glove.stoi:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
                cc = np.mean(emb, axis=0)
                query_sets_1.append(cc)
            else:
                cc = glove.vectors[glove.stoi[lianjie]]
                query_sets_1.append(cc.numpy())
    dd = []
    qq = []
    for i in connect:
        if i not in query_sets:
            hh = eid2name[i]
            hh = hh.split()
            for j in hh:
                if j not in glove.stoi:
                    flag = 0
                    break
                flag = 1
            if flag:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
                qq.append(i)
                aaa = np.mean(emb, axis=0)
                dd.append(aaa)

    scores = cos(query_sets_1, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)

    this_score_ranking = [i
                          for i in score_ranking[:num_qianduoshaoge_2+len(query_sets)]]
    yy = [scores[i]
          for i in score_ranking[:num_qianduoshaoge_2+len(query_sets)]]
    this_keywords1 = [qq[i]
                      for i in score_ranking[:num_qianduoshaoge_2+len(query_sets)]]

    for mm in this_keywords1:
        if mm not in result:
            result.append(mm)

    query_sets_1 = []
    flag = 0
    for i in query_sets:
        hh = eid2name[i]
        hh = hh.split()
        
        for m in hh:
            if m not in glove.stoi:
                flag = 0
                break
            flag = 1
        if flag:
            lianjie = ''.join(hh[:])
            if lianjie not in glove.stoi:
                emb = [glove.vectors[glove.stoi[j]].numpy() for j in hh]
                cc = np.mean(emb, axis=0)
                query_sets_1.append(cc)
            else:
                cc = glove.vectors[glove.stoi[lianjie]]
                query_sets_1.append(cc.numpy())
    dd = []
    qq = []
    for i in connect:
        if i not in query_sets:
            hh = eid2name[i]
            hh = hh.split()
            hh = ''.join(hh[:])
            if hh in glove.stoi:
                cc = glove.vectors[glove.stoi[hh]]
                aaa = cc.numpy()
                qq.append(i)
                dd.append(aaa)
    scores = cos(query_sets_1, dd)
    scores = np.mean(scores[:], axis=0)
    score_ranking = np.argsort(-scores)

    this_score_ranking = [i
                          for i in score_ranking[:num_qianduoshaoge+len(query_sets)]]
    yy = [scores[i]
          for i in score_ranking[:num_qianduoshaoge+len(query_sets)]]
    this_keywords = [qq[i]
                     for i in score_ranking[:num_qianduoshaoge+len(query_sets)]]

    return result+this_keywords


class Expan(object):

    # def __init__(self, args, device, model_name='bert-large-cased-whole-word-masking', dim=1024):
    # def __init__(self, args, device, model_name='roberta-large', dim=1024):

    def __init__(self, args, device, model_name='bert-large-cased-whole-word-masking', dim=1024):
        # self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        # self.maskedLM = RobertaModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=False)

        self.maskedLM = BertForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True)
        self.maskedLM.to(device)
        self.maskedLM.eval()

        self.eid2name, self.keywords, self.eid2idx = load_vocab(
            './apr/entity2id.txt')
        self.entity_pos = pickle.load(
            open('./apr/entity_pos.pkl', 'rb'))

        self.pretrained_emb = np.memmap('./apr/pretrained_emb.npy',
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
        # print(eid_rank[:5])
        return eid_rank[:1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dataset', default='./data/apr', help='path to dataset folder')
    parser.add_argument('-output', default='./apr/results',
                        help='file name for output')
    args = parser.parse_args()

    expan_1 = Expan(args, torch.device("cuda:0"))

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
            expan_glove_1 = pre_expan(query_sets[i], 0.6, 1)
            expan_glove_2 = pre_expan(expan_glove_1, 0.62, 2)
            candidate_pool = candidate(expan_glove_2)
            expanded = expan_1.expand(
                query_sets[i], expan_glove_2, candidate_pool, 50,  gt)
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
