import heapq

import numpy
import ot

from bpe_eval.cal_coverage import get_frequencies, get_frequencies_overlap, tokenize_overlap
from utils.bpe_ot import get_tag2Idx, get_Idx2tag, get_subvocab, get_train_word_label_num, get_test_cost, bpe_normalize, \
    show_top_n_words
from utils.bpe_ot_1 import get_word_label_num, get_rev_dict, word_label_inflect, get_train_bpe_label_num_1, \
    get_test_cost_1, bpe_normalize_1
from utils.bpe_ot_2 import bpe_normalize_2, get_test_cost_2
from utils.data_process import NerProcessor
from tokenizers import MyBertTokenizer
from utils.target_plain import cal_distribution


def get_tokenizer_ot(label_mode, train_path, test_path):
    do_lower_case = False
    bert_name = "bert-base-uncased"

    # code for processor preparing
    processor = NerProcessor(label_mode=label_mode)

    include_bi = True
    include_o = False
    top_num = 200

    # tokenizer
    vocab_file = "../vocab.txt"
    my_tokenizer = MyBertTokenizer(vocab_file, do_lower_case=do_lower_case)
    # tag vocab
    tag2Idx = get_tag2Idx(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    idx2tag = get_Idx2tag(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    print(idx2tag)
    # word vocab
    vocab = my_tokenizer.vocab
    ids_to_tokens = my_tokenizer.ids_to_tokens

    part2full = {}
    full2part = {}

    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    train_examples = processor.get_train_examples(train_path)

    entity_bpe_set_test, label_cnt_test = get_frequencies(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=test_path,
        file_name="train_lexicon.txt",
        include_bi=include_bi,
        include_o=include_o
    )

    entity_bpe_set_train, label_cnt_train = get_frequencies(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=train_path,
        file_name="train.txt",
        include_bi=include_bi,
        include_o=include_o
    )
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_test)
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_train)

    start_point = numpy.zeros((len(tag2Idx),), dtype=numpy.int)
    end_point = numpy.zeros((len(part2full),), dtype=numpy.int)
    cost = numpy.zeros((len(tag2Idx), len(part2full)), dtype=numpy.float)
    get_train_word_label_num(train_examples,
                             my_tokenizer,
                             full2part=full2part,
                             start_point=start_point,
                             end_point=end_point,
                             tag2Idx=tag2Idx,
                             include_bi=include_bi,
                             include_o=include_o)
    M = get_test_cost(entity_bpe_set_test, label_cnt_test, entity_bpe_set_train, cost, full2part=full2part,
                      tag2Idx=tag2Idx, vocab=vocab, take_log=True)

    distribution = ot.sinkhorn(
        start_point,
        end_point,
        M,
        reg=0.8,
        method='sinkhorn',
        numItermax=400
    )  # (6, 5k)

    bpe_normalize(entity_bpe_set_train, distribution, vocab, full2part)
    show_top_n_words(distribution, tag2Idx, idx2tag, ids_to_tokens, part2full=part2full, top_num=top_num)
    # show_top_n_words(part2full=part2full, top_num=top_num)

    tokenizer_train = MyBertTokenizer.from_pretrained(bert_name,
                                                      vocab_file=vocab_file,
                                                      tag2idx=tag2Idx,
                                                      full2part=full2part,
                                                      do_lower_case=do_lower_case,
                                                      word_tokenizer='ot',
                                                      include_bi=include_bi,
                                                      include_o=include_o)
    tokenizer_train.set_M(M=distribution)
    return tokenizer_train


def get_tokenizer_ot1(label_mode, train_path, test_path):
    do_lower_case = False
    bert_name = "bert-base-uncased"

    # code for processor preparing
    processor = NerProcessor(label_mode=label_mode)

    include_bi = True
    include_o = False
    top_num = 200

    # tokenizer
    vocab_file = "../vocab.txt"
    my_tokenizer = MyBertTokenizer(vocab_file, do_lower_case=do_lower_case)
    # tag vocab
    tag2Idx = get_tag2Idx(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    idx2tag = get_Idx2tag(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    print(idx2tag)
    # word vocab
    vocab = my_tokenizer.vocab
    ids_to_tokens = my_tokenizer.ids_to_tokens

    part2full = {}
    full2part = {}

    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    train_examples = processor.get_train_examples(train_path)

    entity_bpe_set_test, label_cnt_test = get_frequencies_overlap(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=test_path,
        file_name="train_lexicon.txt",
        include_bi=include_bi,
        include_o=include_o
    )

    entity_bpe_set_train, label_cnt_train = get_frequencies_overlap(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=train_path,
        file_name="train.txt",
        include_bi=include_bi,
        include_o=include_o
    )
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_test)
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_train)

    word_label_set = get_word_label_num(
        tag2Idx,
        processor,
        vocab,
        tokenizer=my_tokenizer,
        dir_path=train_path,
        file_name="train.txt",
        include_bi=True,
        include_o=False
    )
    total_len, word_part2idx, idx2label_part, label_part2idx, word_idxes = word_label_inflect(tag2Idx, word_label_set)
    # word_idxes : {startpoint: length}
    idx2word_part = get_rev_dict(word_part2idx)

    # print(word_label_set)

    word_bpe_mask = numpy.zeros((total_len, len(part2full)), dtype=numpy.int)
    for word in word_part2idx:
        # print(word)
        subword_list = tokenize_overlap(list(word), vocab)
        # print(subword_list)
        for subword in subword_list:
            subword_id = full2part[vocab[subword]]
            word_id = word_part2idx[word]
            label_num = word_idxes[word_id]
            word_bpe_mask[word_id: word_id + label_num, subword_id] = 1

    start_point = numpy.zeros((total_len,), dtype=numpy.int)
    end_point = numpy.zeros((len(part2full),), dtype=numpy.int)
    cost = numpy.zeros((total_len, len(part2full)), dtype=numpy.float)
    get_train_bpe_label_num_1(train_examples,
                              my_tokenizer,
                              word_idxes,
                              word_part2idx,
                              idx2label_part,
                              full2part=full2part,
                              start_point=start_point,
                              end_point=end_point,
                              tag2Idx=tag2Idx,
                              include_bi=include_bi,
                              include_o=include_o)
    M = get_test_cost_1(
        entity_bpe_set_test,
        label_cnt_test,
        entity_bpe_set_train,
        word_bpe_mask,
        label_part2idx,
        cost,
        full2part=full2part,
        tag2Idx=tag2Idx,
        vocab=vocab,
        take_log=True
    )
    print(M.shape)

    distribution = ot.sinkhorn(
        start_point,
        end_point,
        M,
        reg=0.8,
        method='sinkhorn',
        numItermax=500
    )  # (6, 5k)

    # print(distribution[0, :])

    bpe_normalize_1(distribution, vocab, full2part, idx2word_part)

    # max_idx = numpy.argmax(distribution, axis=1)
    cur_word = ""
    for i in range(10):
        l = list(distribution[i])
        l = enumerate(l)
        if i in idx2word_part:
            cur_word = idx2word_part[i]
            print("cur word: ", cur_word, end=" || ")
        print("cur label: ", idx2label_part[i])
        topn = heapq.nlargest(top_num, l, key=lambda x: x[1])
        topn_score = [x[1] for x in topn]
        topn = [ids_to_tokens[part2full[x[0]]] for x in topn]
        print(topn)
        print(topn_score)

    # show_top_n_words(distribution, tag2Idx, idx2tag, ids_to_tokens, part2full=part2full, top_num=top_num)
    # # show_top_n_words(part2full=part2full, top_num=top_num)
    #
    tokenizer_train = MyBertTokenizer.from_pretrained(bert_name,
                                                      vocab_file=vocab_file,
                                                      tag2idx=tag2Idx,
                                                      full2part=full2part,
                                                      word_part2idx=word_part2idx,
                                                      idx2label_part=idx2label_part,
                                                      word_idxes=word_idxes,
                                                      do_lower_case=do_lower_case,
                                                      word_tokenizer='ot1',
                                                      include_bi=include_bi,
                                                      include_o=include_o)
    tokenizer_train.set_M(M=distribution)
    return tokenizer_train


def get_tokenizer_ot2(label_mode, train_path, test_path):
    do_lower_case = False
    bert_name = "bert-base-uncased"

    # code for processor preparing
    processor = NerProcessor(label_mode=label_mode)

    include_bi = True
    include_o = False
    top_num = 200

    # tokenizer
    vocab_file = "../vocab.txt"
    my_tokenizer = MyBertTokenizer(vocab_file, do_lower_case=do_lower_case)
    # tag vocab
    tag2Idx = get_tag2Idx(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    idx2tag = get_Idx2tag(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    print(idx2tag)
    # word vocab
    vocab = my_tokenizer.vocab
    ids_to_tokens = my_tokenizer.ids_to_tokens

    part2full = {}
    full2part = {}

    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    train_examples = processor.get_train_examples(train_path)

    entity_bpe_set_test, label_cnt_test = get_frequencies_overlap(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=test_path,
        file_name="train_lexicon.txt",
        include_bi=include_bi,
        include_o=include_o
    )

    entity_bpe_set_train, label_cnt_train = get_frequencies_overlap(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=train_path,
        file_name="train.txt",
        include_bi=include_bi,
        include_o=include_o
    )
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_test)
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_train)

    word_label_set = get_word_label_num(
        tag2Idx,
        processor,
        vocab,
        tokenizer=my_tokenizer,
        dir_path=train_path,
        file_name="train.txt",
        include_bi=True,
        include_o=False
    )
    total_len, word_part2idx, idx2label_part, label_part2idx, word_idxes = word_label_inflect(tag2Idx, word_label_set)
    # word_idxes : {startpoint: length}
    idx2word_part = get_rev_dict(word_part2idx)

    # print(word_label_set)

    word_bpe_mask = numpy.zeros((total_len, len(part2full)), dtype=numpy.int)
    for word in word_part2idx:
        # print(word)
        subword_list = tokenize_overlap(list(word), vocab)
        # print(subword_list)
        for subword in subword_list:
            subword_id = full2part[vocab[subword]]
            word_id = word_part2idx[word]
            label_num = word_idxes[word_id]
            word_bpe_mask[word_id: word_id + label_num, subword_id] = 1

    start_point = numpy.zeros((total_len,), dtype=numpy.int)
    end_point = numpy.zeros((len(part2full),), dtype=numpy.int)
    cost = numpy.zeros((total_len, len(part2full)), dtype=numpy.float)
    get_train_bpe_label_num_1(train_examples,
                              my_tokenizer,
                              word_idxes,
                              word_part2idx,
                              idx2label_part,
                              full2part=full2part,
                              start_point=start_point,
                              end_point=end_point,
                              tag2Idx=tag2Idx,
                              include_bi=include_bi,
                              include_o=include_o)
    M = get_test_cost_2(
        entity_bpe_set_test,
        label_cnt_test,
        entity_bpe_set_train,
        label_cnt_train,
        word_bpe_mask,
        label_part2idx,
        cost,
        full2part=full2part,
        tag2Idx=tag2Idx,
        vocab=vocab,
        take_log=True
    )
    print(M.shape)

    distribution = ot.sinkhorn(
        start_point,
        end_point,
        M,
        reg=0.8,
        method='sinkhorn',
        numItermax=500
    )  # (6, 5k)

    # print(distribution[0, :])

    bpe_normalize_2(distribution, word_bpe_mask)

    # max_idx = numpy.argmax(distribution, axis=1)
    cur_word = ""
    for i in range(10):
        l = list(distribution[i])
        l = enumerate(l)
        if i in idx2word_part:
            cur_word = idx2word_part[i]
            print("cur word: ", cur_word, end=" || ")
        print("cur label: ", idx2label_part[i])
        topn = heapq.nlargest(top_num, l, key=lambda x: x[1])
        topn_score = [x[1] for x in topn]
        topn = [ids_to_tokens[part2full[x[0]]] for x in topn]
        print(topn)
        print(topn_score)

    # show_top_n_words(distribution, tag2Idx, idx2tag, ids_to_tokens, part2full=part2full, top_num=top_num)
    # # show_top_n_words(part2full=part2full, top_num=top_num)
    #
    tokenizer_train = MyBertTokenizer.from_pretrained(bert_name,
                                                      vocab_file=vocab_file,
                                                      tag2idx=tag2Idx,
                                                      full2part=full2part,
                                                      word_part2idx=word_part2idx,
                                                      idx2label_part=idx2label_part,
                                                      word_idxes=word_idxes,
                                                      do_lower_case=do_lower_case,
                                                      word_tokenizer='ot1',
                                                      include_bi=include_bi,
                                                      include_o=include_o)
    tokenizer_train.set_M(M=distribution)
    return tokenizer_train


def get_tokenizer_target(label_mode, train_path, test_path):
    do_lower_case = False
    bert_name = "bert-base-uncased"

    # code for processor preparing
    processor = NerProcessor(label_mode=label_mode)

    include_bi = True
    include_o = False
    top_num = 200

    # tokenizer
    vocab_file = "../vocab.txt"
    my_tokenizer = MyBertTokenizer(vocab_file, do_lower_case=do_lower_case)
    # tag vocab
    tag2Idx = get_tag2Idx(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    idx2tag = get_Idx2tag(processor.get_labels(), include_bi=include_bi, include_o=include_o)
    print(idx2tag)
    # word vocab
    vocab = my_tokenizer.vocab
    ids_to_tokens = my_tokenizer.ids_to_tokens

    part2full = {}
    full2part = {}

    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    train_examples = processor.get_train_examples(train_path)

    entity_bpe_set_test, label_cnt_test = get_frequencies_overlap(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=test_path,
        file_name="train_lexicon.txt",
        include_bi=include_bi,
        include_o=include_o
    )

    entity_bpe_set_train, label_cnt_train = get_frequencies_overlap(
        tag2Idx,
        processor,
        tokenizer=my_tokenizer,
        dir_path=train_path,
        file_name="train.txt",
        include_bi=include_bi,
        include_o=include_o
    )
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_test)
    get_subvocab(vocab, part2full, full2part, entity_bpe_set_train)

    distribution = numpy.zeros((len(tag2Idx), len(part2full)), dtype=numpy.float)

    cal_distribution(
        entity_bpe_set_test,
        label_cnt_test,
        vocab,
        tag2Idx,
        distribution,
        full2part
    )
    tokenizer_train = MyBertTokenizer.from_pretrained(bert_name,
                                                      vocab_file=vocab_file,
                                                      tag2idx=tag2Idx,
                                                      full2part=full2part,
                                                      do_lower_case=do_lower_case,
                                                      word_tokenizer='ot',
                                                      include_bi=include_bi,
                                                      include_o=include_o)
    tokenizer_train.set_M(M=distribution)
    return tokenizer_train



