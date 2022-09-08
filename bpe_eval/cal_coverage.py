import heapq
import sys
import os

from transformers import BertTokenizer

sys.path.append(os.path.abspath("../.."))
from collections import defaultdict


def expand_length_1(length: int):
    if length == 2 or length == 3:
        return length + 1
    return length


def expand_length_2(length: int):
    if length == 2:
        return length + 1
    return length


def expand_length_3(length: int):
    if length <= 4:
        return length + 1
    return length


def expand_length_4(length: int):
    if length == 2 or length == 3:
        return length + 1
    return length


def expand_length_5(length: int):
    if length == 2 or length == 3:
        return length + 1
    return length


def expand_length_6(length: int):
    # return length // 2 + 2
    if length == 2 or length == 3:
        return length + 1
    return length


def gen_top_num(length):
    if length <= 2:
        return length * 2
    elif length <= 4:
        return length * 2 + 1
    else:
        return 9


def subseq_score(subseq, target_token_label_count, label):
    score = 1.0
    for subtoken in subseq:
        if subtoken not in target_token_label_count:
            score = score * 1e-9
        elif label not in target_token_label_count[subtoken]:
            score = score * 1e-9 / sum(target_token_label_count[subtoken].values())
        else:
            score = score * target_token_label_count[subtoken][label] / sum(target_token_label_count[subtoken].values())
    return score


def subseq_score_2(subseq, A, B):
    score = 0.0
    length = len(subseq)
    for subtoken in subseq:
        score += len(subtoken)
    score /= length
    return score


def get_top_n_subseqs(subseqs, target_token_label_count, label, top_num=5, score_function=subseq_score):
    subseqs_scores = enumerate([score_function(subseq, target_token_label_count, label) for subseq in subseqs])
    topn = heapq.nlargest(top_num, subseqs_scores, key=lambda x: x[1])
    # topn_score = [x[1] for x in topn]
    topn = [subseqs[x[0]] for x in topn]
    return topn


def tokenize_overlap(chars, vocab):
    start = 0
    sub_tokens = []
    while start < len(chars):
        end = len(chars)
        while start < end:
            substr = "".join(chars[start:end])
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                sub_tokens.append(substr)
            end -= 1
        start += 1
    if sub_tokens == []:
        return ["[UNK]"]
    return sub_tokens


def tokenize_overlap_count(chars, vocab, middle=False, max_token_num=5, depth=0, max_depth=6, mode="cur",
                           tokenizer=None,
                           target_token_label_count=None, label=None):
    # if middle == False:
    #     print(chars)
    if depth > max_depth:
        return []
    sub_tokens = []  # 一个列表，里面每项是一个分法的所有subtoken的列表

    end = 1
    while end <= len(chars):
        substr = "".join(chars[:end])
        if middle:
            substr = "##" + substr
        if substr in vocab:
            sub_sub_tokens = tokenize_overlap_count(chars[end:], vocab, middle=True,
                                                    max_token_num=max_token_num, depth=depth + 1, max_depth=max_depth)
            if sub_sub_tokens:
                for i in range(len(sub_sub_tokens)):
                    sub_sub_tokens[i].append(substr)
            elif end == len(chars):
                sub_sub_tokens = [[substr]]
            else:
                sub_sub_tokens = []
            sub_tokens += sub_sub_tokens
        end += 1

    if middle:
        return sub_tokens
    else:
        if mode == "func1":
            sub_tokens = [l for l in sub_tokens if expand_length_1(len(sub_tokens[-1])) >= len(l)]
        elif mode == "func2":
            sub_tokens = [l for l in sub_tokens if expand_length_2(len(sub_tokens[-1])) >= len(l)]
        elif mode == "func3":
            sub_tokens = [l for l in sub_tokens if expand_length_2(len(sub_tokens[-1])) >= len(l)]
            top_num = min(len(sub_tokens), 15)
            sub_tokens = get_top_n_subseqs(sub_tokens, target_token_label_count, label, top_num=top_num,
                                           score_function=subseq_score_2)
        elif mode == "func4":
            sub_tokens = [l for l in sub_tokens if expand_length_3(len(sub_tokens[-1])) >= len(l)]
            if sub_tokens:
                top_num = gen_top_num(len(sub_tokens[-1]))
                top_num = min(len(sub_tokens), top_num)
                sub_tokens = get_top_n_subseqs(sub_tokens, target_token_label_count, label, top_num=top_num,
                                               score_function=subseq_score_2)
        elif mode == "func5":
            top_num = min(len(sub_tokens), 3)
            sub_tokens = get_top_n_subseqs(sub_tokens, target_token_label_count, label, top_num=top_num)
            sub_tokens = [l for l in sub_tokens if expand_length_3(len(sub_tokens[-1])) >= len(l)]
        elif mode == "func6":
            top_num = min(len(sub_tokens), 4)
            sub_tokens = get_top_n_subseqs(sub_tokens, target_token_label_count, label, top_num=top_num)
            sub_tokens = [l for l in sub_tokens if expand_length_1(len(sub_tokens[-1])) >= len(l)]
        else:
            sub_tokens = [l for l in sub_tokens if len(l) < max_token_num]
        sub_tokens_count = defaultdict(int)
        sub_tokens_dict = []
        if not sub_tokens:
            sub_tokens.append(tokenizer.tokenize("".join(chars)))
        for i, subseq in enumerate(sub_tokens):
            token_dict = defaultdict(int)
            for subtoken in subseq:
                sub_tokens_count[subtoken] += 1
                token_dict[subtoken] += 1
            sub_tokens_dict.append(token_dict)
        return sub_tokens_count, sub_tokens, sub_tokens_dict


def cal_EECR(
        entity_bpe_set_test,
        label_cnt_test,
        entity_bpe_set_train,
        label_cnt_train):
    sum_ecr = 0
    for bpe in entity_bpe_set_train:
        if bpe not in entity_bpe_set_test:
            continue
        sum_classes = 0
        sum_frq = entity_bpe_set_train[bpe]
        for label in label_cnt_train:
            if bpe in label_cnt_train[label] and bpe in label_cnt_test[label]:
                p_train = float(label_cnt_train[label][bpe]) / sum_frq
                c_test = label_cnt_test[label][bpe]
                sum_classes += p_train * c_test
        # sum_classes /= entity_bpe_set_test[bpe]
        sum_ecr += sum_classes
    sum_ecr /= sum(entity_bpe_set_test.values())
    return sum_ecr


def cal_EECR_plain(
        entity_bpe_set_test,
        label_cnt_test,
        entity_bpe_set_train,
        label_cnt_train):
    sum_ecr = 0
    for bpe in entity_bpe_set_train:
        if bpe not in entity_bpe_set_test:
            continue
        sum_classes = 0
        sum_train_frq = entity_bpe_set_train[bpe]
        sum_test_frq = entity_bpe_set_test[bpe]
        for label in label_cnt_train:
            if bpe in label_cnt_train[label] and bpe in label_cnt_test[label]:
                p_train = float(label_cnt_train[label][bpe]) / sum_train_frq
                p_test = float(label_cnt_test[label][bpe]) / sum_test_frq
                sum_classes += p_train * p_test
        # sum_classes /= entity_bpe_set_test[bpe]
        sum_ecr += sum_classes
    sum_ecr /= len(entity_bpe_set_test)
    return sum_ecr


if __name__ == '__main__':
    vocab = {"b", "ba", "##na", "##a", "banana", "##nan"}
    bert_name = "bert-base-cased"
    ori_tokenizer = BertTokenizer.from_pretrained(
        bert_name, do_lower_case=False
    )
    a, b, c = tokenize_overlap_count(list("banana"), vocab, False, tokenizer=ori_tokenizer)
    print(a)
    print(b)
    print(c)
