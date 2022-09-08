# coding=utf-8
import logging
import time
from math import log2
import sys
import os
sys.path.append(os.path.abspath(".."))
from mytokenizers  import MyBertTokenizer
from bpe_eval.cal_coverage import tokenize_overlap_count
from utils.data_process import NerProcessor, InputFeatures
import ot
import numpy as np
from collections import defaultdict
from transformers import BertTokenizer
import copy
import scipy.stats
import heapq


def get_rev_dict(mydict):
    return {v: k for k, v in mydict.items()}


def get_word_token_label_statistics(
        processor,
        tokenizer,
        dir_path=None,
        file_name="train.txt",
        max_token_num=5,
        mode="cur",
        target_token_label_count=None
):

    examples = processor.get_test_examples(
        dir_path,
        file_name
    )
    word_label_count = defaultdict(dict)
    token_label_count = defaultdict(dict)
    word2subtoken = {}
    word2subtokenlist = {}
    word2subtokendict = {}
    token_count = {}
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label

        for label, word in zip(labellist, textlist):
            # label=O也一起统计了再说
            if label != "O":
                label = label[2:] # 去掉BI

            word_pieces = tokenizer.basic_tokenizer.tokenize(word, never_split=tokenizer.all_special_tokens)
            for w in word_pieces:
                if w not in word2subtoken:
                    chars = list(w)
                    subtoken_count, subtoken_list, subtoken_dict = tokenize_overlap_count(chars, tokenizer.vocab, max_token_num=max_token_num,
                                            mode=mode, tokenizer=tokenizer, target_token_label_count=target_token_label_count, label=label) # 当前word可能tokenize的所有subtoken
                    word2subtoken[w] = subtoken_count
                    word2subtokenlist[w] = subtoken_list
                    word2subtokendict[w] = subtoken_dict
                for token, count in word2subtoken[w].items():
                    token_label_count[token][label] = token_label_count[token].get(label, 0) + count
                word_label_count[w][label] = word_label_count[w].get(label, 0) + 1

    return word_label_count, word2subtoken, word2subtokenlist, word2subtokendict, token_label_count



# 需要换一个tokenizer
def get_token_label_statistics(
        processor,
        tokenizer,
        dir_path=None,
        file_name="train.txt",
        use_label=False,
        count_O=True
):

    examples = processor.get_test_examples(
        dir_path,
        file_name
    )

    token_label_count = defaultdict(dict)
    token_total = 0

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label

        for ori_label, word in zip(labellist, textlist):
            # label=O也一起统计了再说
            if ori_label != "O":
                label = ori_label[2:] # 去掉BI
            else:
                label = ori_label

            if use_label:
                word_tokens = tokenizer.ot_tokenize(word, ori_label)
            else:
                word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                for token in word_tokens:
                    token_label_count[token][label] = token_label_count[token].get(label, 0) + 1

            # count_O: 是否统计label为O的词
            # count_O = True
            if count_O:
                token_total += len(word_tokens)
            elif label != "O":
                token_total += len(word_tokens)

    return token_label_count, token_total


def get_start_end_points(word_label_count, word2subtoken, word_label2id, token2id, entity_word_set, entity_label_list):
    start_point = np.zeros((len(word_label2id),), dtype=np.int)
    end_point = np.zeros((len(token2id),), dtype=np.int)

    for word in entity_word_set:
        for label in entity_label_list:
            word_label_idx = word_label2id[word+"_"+label]
            if label not in word_label_count[word]:
                continue
            for token, token_num in word2subtoken[word].items():
                token_idx = token2id[token]
                # label_token_num = word_label_count[word].get(label,0) * token_num  # 该word为该label的次数 * token被分出的次数
                label_token_num = word_label_count[word].get(label, 0)   # 每个包含的token只加一次，word总数是出现次数*包含的token数
                end_point[token_idx] += label_token_num
                start_point[word_label_idx] += label_token_num  # 每个token的次数都累加，得到word_label的次数

    return start_point, end_point


def cal_subseq_ratio(distribution, word2subtokenlist, word2subtokendict, word_label2id,
                                               token2id, entity_word_set, entity_label_list):

    # word2subtokenlist: {word1: [], word2: []} 列表中每一项都是一个分法的subword list

    word2subtokenlist_ratio = defaultdict(dict)
    # {word1:{label1:[], label2:[]...}, word2:{}..}
    # 每个标签下是一个和subtokenlist顺序对应的概率列表，每一项为对应subtokenlist在该标签下出现的概率。每个标签的所有分法概率归一化

    for word in entity_word_set:
        for label in entity_label_list:
            word_label_idx = word_label2id[word+"_"+label]
            word_label_to_token_distrib = distribution[word_label_idx]  # 取出该行的分布

            if abs(word_label_to_token_distrib.sum() - 0 ) < 1e-10:
                subseq_ratio = np.zeros((len(word2subtokenlist[word]),), dtype=np.float)

            else:
                subtokenlist = word2subtokenlist[word]
                subseq_ratio = np.zeros((len(subtokenlist),), dtype=np.float)
                for idx, subseq in enumerate(subtokenlist):
                    ids = [token2id[token] for token in subseq]
                    scores = word_label_to_token_distrib[ids]
                    subseq_ratio[idx] = scores.min()  # 以每个序列中最小的token频数作为该切分的频数
                    # if np.around(subseq_ratio[idx]) < 1: #word_label_count[word][label]:  # word_label_count[word][label]: # 1 is a threshold, or can set to word_label count
                    #     subseq_ratio[idx] = 0
                if subseq_ratio.sum() != 0:
                    subseq_ratio = subseq_ratio / subseq_ratio.sum()

            word2subtokenlist_ratio[word][label] = copy.deepcopy(subseq_ratio.tolist()) # 需要deepcopy吗

    return word2subtokenlist_ratio



def cal_subseq_ratio_top1(distribution, word2subtokenlist, word2subtokendict, word_label2id,
                                               token2id, entity_word_set, entity_label_list):

    # word2subtokenlist: {word1: [], word2: []} 列表中每一项都是一个分法的subword list

    word2subtokenlist_ratio = defaultdict(dict)
    # {word1:{label1:[], label2:[]...}, word2:{}..}
    # 每个标签下是一个和subtokenlist顺序对应的概率列表，每一项为对应subtokenlist在该标签下出现的概率。每个标签的所有分法概率归一化

    for word in entity_word_set:
        for label in entity_label_list:
            word_label_idx = word_label2id[word+"_"+label]
            word_label_to_token_distrib = distribution[word_label_idx]  # 取出该行的分布

            if abs(word_label_to_token_distrib.sum() - 0 ) < 1e-10:
                subseq_ratio = np.zeros((len(word2subtokenlist[word]),), dtype=np.float)

            else:
                subtokenlist = word2subtokenlist[word]
                subseq_ratio = np.zeros((len(subtokenlist),), dtype=np.float)
                min_idx = -1
                min_scores = None
                for idx, subseq in enumerate(subtokenlist):
                    ids = [token2id[token] for token in subseq]
                    scores = word_label_to_token_distrib[ids]
                    score = scores.min() # 以每个序列中最小的token频数作为该切分的频数
                    if min_idx < 0 or (min_idx >= 0 and score >= subseq_ratio[min_idx]):
                        if min_idx >= 0:
                            subseq_ratio[min_idx] = 0.
                        min_idx = idx
                        subseq_ratio[idx] = score
                        min_scores = scores

                if subseq_ratio[min_idx] < word_label_count[word][label]: #word_label_count[word][label]: # 1 is a threshold, or can set to word_label count
                    subseq_ratio[min_idx] = 0
                if subseq_ratio.sum() != 0:
                    subseq_ratio = subseq_ratio / subseq_ratio.sum()

            word2subtokenlist_ratio[word][label] = copy.deepcopy(subseq_ratio.tolist()) # 需要deepcopy吗

    return word2subtokenlist_ratio


def cal_subseq_ratio_2(distribution, word2subtokenlist, word2subtokendict, word_label2id, token2id, entity_word_set, entity_label_list):

    # word2subtokenlist: {word1: [], word2: []} 列表中每一项都是一个分法的subword list

    word2subtokenlist_ratio = defaultdict(dict)
    # {word1:{label1:[], label2:[]...}, word2:{}..}
    # 每个标签下是一个和subtokenlist顺序对应的概率列表，每一项为对应subtokenlist在该标签下出现的概率。每个标签的所有分法概率归一化

    for word in entity_word_set:
        for label in entity_label_list:
            word_label_idx = word_label2id[word+"_"+label]
            word_label_to_token_distrib = distribution[word_label_idx]  # 取出该行的分布

            if abs(word_label_to_token_distrib.sum() - 0 ) < 1e-10:
                split_num = np.zeros((len(word2subtokenlist[word]),), dtype=np.float)

            else:
                # word2subtokendict: 把word2subtokenlist的每项改成一个词典：{token:token_num}
                subtokenlist = word2subtokenlist[word]
                subtokendict = word2subtokendict[word]
                subtokens = word2subtoken[word].keys()
                subtoken_num = np.zeros((len(subtokens),), dtype=np.float)
                split_matrix = np.zeros((len(subtokens), len(subtokenlist)), dtype=np.float)

                # 这里没有对token数目做归一化，不过貌似也可以不用
                for idx, token in enumerate(subtokens):
                    token_idx = token2id[token]
                    subtoken_num[idx] = word_label_to_token_distrib[token_idx]
                    for s_idx, subdict in enumerate(subtokendict):
                        split_matrix[idx, s_idx] = subdict.get(token,0)

                split_num = np.linalg.pinv(split_matrix).dot(subtoken_num)
                split_num = np.maximum(0, split_num)
                split_num = split_num/split_num.sum()

            word2subtokenlist_ratio[word][label] = copy.deepcopy(split_num.tolist()) # 需要deepcopy吗

    return word2subtokenlist_ratio



def cal_KL_score(source_token_label_count, target_token_label_count, source_token_total, target_token_total, entity_label_list):
    # 计算KL散度
    # smoothing 方法好像不太对
    KL_score = 0.
    token_set = source_token_label_count.keys() | target_token_label_count.keys()
    for token in token_set:
        source_frq = np.zeros(len(entity_label_list), dtype=np.float)
        target_frq = np.zeros(len(entity_label_list), dtype=np.float)
        for idx, label in enumerate(entity_label_list):
            source_frq[idx] = source_token_label_count[token].get(label, 1e-20)
            target_frq[idx] = target_token_label_count[token].get(label, 1e-20)

        entropy = scipy.stats.entropy(source_frq, target_frq)
        KL_score += entropy

    return KL_score/len(token_set)


def cal_KL_score_conditional(source_token_label_count, target_token_label_count, source_token_total, target_token_total, entity_label_list):
    # 计算KL散度
    # smoothing 方法好像不太对
    KL_score = 0.
    token_set = source_token_label_count.keys() | target_token_label_count.keys()
    # entity_label_list = entity_label_list + ["O"] 加还是不加呢？求的时候好像是算上的
    for token in token_set:
        source_frq = np.zeros(len(entity_label_list), dtype=np.float)
        target_frq = np.zeros(len(entity_label_list), dtype=np.float)
        for idx, label in enumerate(entity_label_list):
            source_frq[idx] = source_token_label_count[token].get(label, 1e-20)
            target_frq[idx] = target_token_label_count[token].get(label, 1e-20)
        source_frq = source_frq / source_frq.sum()
        target_frq = target_frq / target_frq.sum()
        entropy = scipy.stats.entropy(source_frq, target_frq)
        KL_score += entropy

    return KL_score/len(token_set)


import argparse

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_mode", type=str, default='ontonotes')
    parser.add_argument("--source_domain",type=str,default='bn')
    parser.add_argument("--target_domain",type=str,default='mz')
    parser.add_argument("--source_path", type=str, default='bn')
    parser.add_argument("--target_path", type=str, default='mz')
    parser.add_argument("--subword_data_dir",type=str,default='/opt/tiger/bpe_ot/ot_test/test/')
    parser.add_argument("--mode",type=str,default='mxlen')
    parser.add_argument("--reg",type=float,default=1.0)
    parser.add_argument("--max_token_num",type=int,default=5)
    parser.add_argument("--log_file",type=str,default="log.log")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_parser()
    logger = logging.getLogger(__name__)
    os.environ["WANDB_DISABLED"] = "true"
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)
    bert_name = "bert-base-cased"
    domain_list = ["bc", "bn", "mz", "nw", "tc", "wb"]
    label_mode = args.label_mode
    source_domain = args.source_domain
    target_domain = args.target_domain
    source_path = args.source_path
    target_path = args.target_path
    subword_data_dir = args.subword_data_dir

    print("Calculating ot from {} to {}".format(source_domain, target_domain))
    train_path = source_path
    test_path = target_path

    # code for processor preparing
    processors = {"ner": NerProcessor(label_mode=label_mode)}
    processor = processors["ner"]

    include_bi = True
    include_o = False
    top_num = 20

    # tokenizer
    vocab_file = "../vocab.txt"
    # my_tokenizer = MyBertTokenizer(vocab_file, do_lower_case=do_lower_case)
    ori_tokenizer = BertTokenizer.from_pretrained(
        bert_name, do_lower_case=False
    )



    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    # ignore_labels = ["CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"]
    ignore_labels = []

    entity_label_list = [label[2:] for label in label_list if label.startswith("I-") and label[2:] not in ignore_labels]
    train_examples = processor.get_train_examples(train_path)

    target_file_name = "train_lexicon.txt"
    if target_domain == "conll":
        target_file_name = "train.txt"
    target_token_label_count, target_token_total = get_token_label_statistics(
        processor,
        tokenizer=ori_tokenizer,
        dir_path=test_path,
        file_name=target_file_name,
    )

    word_label_count, word2subtoken, word2subtokenlist, word2subtokendict, token_label_count = get_word_token_label_statistics(
        processor,
        tokenizer=ori_tokenizer,
        dir_path=train_path,
        file_name="train.txt",
        max_token_num=args.max_token_num,
        mode=args.mode,
        target_token_label_count=target_token_label_count
    )

    source_token_label_count, source_token_total = get_token_label_statistics(
        processor,
        tokenizer=ori_tokenizer,
        dir_path=train_path,
        file_name="train.txt"
    )

    # 只考虑entity下的subword分配，因此entity_word_set为去除非entity词的word集合
    entity_word_set = set()
    for word, label_count in word_label_count.items():
        # if len(label_count) == 1: # 只有O
        #     continue
        if len(label_count) == 1 and "O" in label_count: # 只有O
            continue
        entity_word_set.add(word)

    # word_label 到 idx 映射
    word_label2id = {}
    for word in entity_word_set:
        for label in entity_label_list:
            word_label2id[word+"_"+label] = len(word_label2id)
    id2word_label = {value:key for key, value in word_label2id.items()}

    # token 到 idx 映射
    token2id = {}
    for word in entity_word_set:
        for token in word2subtoken[word].keys():
            if token not in token2id:
                token2id[token] = len(token2id)
    id2token = {value:key for key, value in token2id.items()}

    # 构造word_token distribution的mask，对每个word，只有属于该word的token才为1，其余为0
    transport_mask = np.zeros((len(word_label2id), len(token2id)), dtype=np.float)
    for word in entity_word_set:
        for token in word2subtoken[word].keys():
            word_label_id_start = word_label2id[word+"_"+entity_label_list[0]]
            transport_mask[word_label_id_start:word_label_id_start+len(entity_label_list), token2id[token]] = 1.

    start_point, end_point = get_start_end_points(word_label_count, word2subtoken, word_label2id,
                                                  token2id, entity_word_set, entity_label_list)

    cost = np.zeros((len(word_label2id), len(token2id)), dtype=np.float)
    for word in entity_word_set:
        for label in entity_label_list:
            word_label_idx = word_label2id[word+"_"+label]
            for token, token_idx in token2id.items():
                if label not in target_token_label_count[token]:
                    # cost[word_label_idx, token_idx] = np.inf
                    # pass
                    # target 没有这个token
                    if len(target_token_label_count[token]) == 0:
                        cost[word_label_idx, token_idx] = -log2(1e-9)
                    else:
                        # 惩罚target中有这个token且label 不为当前label的项
                        cost[word_label_idx, token_idx] = -1. * log2(1e-9 /
                                                             sum(target_token_label_count[token].values()))
                else: # 省略mask

                    # 推导KL(P(Y|T))得到的cost
                    cost[word_label_idx, token_idx] = -1. * log2(target_token_label_count[token][label] /
                                                             sum(target_token_label_count[token].values()))\
                                                      / max(sum(source_token_label_count[token].values()), 1)
                    # 初始cost KL(P(Y,T))
                    # cost[word_label_idx, token_idx] = -log2(target_token_label_count[token][label]/target_token_total)

    # cost = cost/cost.max()
    cost = cost + (1. - transport_mask) * 100.

    print(cost.shape)
    start_time = time.time()
    distribution = ot.sinkhorn(
        start_point,
        end_point,
        cost,
        reg=args.reg,
        method='sinkhorn',
        numItermax=500
    )  # (6, 5k)
    time_span = time.time() - start_time
    start_point_len = len(start_point)
    end_point_len = len(end_point)

    word2subtokenlist_ratio = cal_subseq_ratio(distribution, word2subtokenlist, word2subtokendict, word_label2id,
                                               token2id, entity_word_set, entity_label_list)



    tokenizer_train = MyBertTokenizer.from_pretrained(bert_name,
                                                      vocab_file=vocab_file,
                                                      do_lower_case=False,
                                                      word_tokenizer='ot',word2subtokenlist=word2subtokenlist,
                                                      word2subtokenlist_ratio=word2subtokenlist_ratio)


    source_token_label_count_ot, source_token_total_ot = get_token_label_statistics(
        processor,
        tokenizer=tokenizer_train,
        dir_path=train_path,
        file_name="train.txt",
        use_label=True
    )

    KL_score = cal_KL_score_conditional(source_token_label_count, target_token_label_count, source_token_total, target_token_total, entity_label_list)
    print("KL score before OT: {}".format(KL_score))
    logger.info("KL score before OT: {}".format(KL_score))

    KL_score = cal_KL_score_conditional(source_token_label_count_ot, target_token_label_count, source_token_total_ot, target_token_total, entity_label_list)
    print("KL score after OT: {}".format(KL_score))
    logger.info("KL score after OT: {}".format(KL_score))
    # tokenizer_train.dump_split_ratio(f"/opt/tiger/bpe_ot/ot_data/{source_domain}2{target_domain}")

    tokenizer_train.dump_split_ratio(args.subword_data_dir)
