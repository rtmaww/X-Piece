import random

from transformers import BertTokenizer
from tokenization.tokenize_1 import tokenize_with_dp_1
import numpy as np

import json


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def tokenize_plain(chars, vocab):
    start = 0
    sub_tokens = []
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            return ["[UNK]"]
        sub_tokens.append(cur_substr)
        start = end
    return sub_tokens


def search_part_score(M, label_id, full2part, fullid):
    if fullid in full2part:
        return M[label_id, full2part[fullid]]
    else:
        return 1e-16


def tokenize_with_dp(chars, label, tag2idx, M, vocab, full2part):
    dp_sub_tokens = {}
    dp_scores = {}
    start_mid = 0
    # if chars[0] not in vocab:
    #     return ["[UNK]"]
    for i in range(len(chars)):
        substr = "".join(chars[0:i + 1])
        if substr in vocab:
            dp_sub_tokens[i] = [substr]
            dp_scores[i] = search_part_score(M, tag2idx[label], full2part, vocab[substr])
            start_mid = i
            break
    if dp_scores == {}:
        return ["[UNK]"]
    # print(dp_sub_tokens)
    cur_score = -1e9
    cur_mid = start_mid
    for end in range(1, len(chars)):
        full_substr = "".join(chars[0:end + 1])
        # print(full_substr)
        if full_substr in vocab:
            cur_score = search_part_score(M, tag2idx[label], full2part, vocab[full_substr])
        else:
            cur_score = -1e9
        # print(cur_score)
        cur_mid = -1
        for mid in range(start_mid, end):
            if mid not in dp_scores:
                continue
            substr = "##" + "".join(chars[mid + 1:end + 1])
            if substr in vocab and dp_scores[mid] * search_part_score(M, tag2idx[label], full2part,
                                                                      vocab[substr]) > cur_score:
                # print(cur_score, "-->", dp_scores[mid], "*", search_part_score(M, tag2idx[label], full2part, vocab[substr]))
                # if cur_mid != -1:
                #     tmp_cur_mid = dp_sub_tokens[cur_mid]
                # else:
                #     tmp_cur_mid = ""
                # print(
                #       tmp_cur_mid,
                #       ", ##", "".join(chars[cur_mid + 1:end + 1]),
                #       "-->",
                #       dp_sub_tokens[mid], ", ##", "".join(chars[mid + 1:end + 1]))
                cur_score = dp_scores[mid] * search_part_score(M, tag2idx[label], full2part, vocab[substr])
                cur_mid = mid
        if cur_score != -1e9:
            dp_scores[end] = cur_score
            if cur_mid != -1:
                dp_sub_tokens[end] = [bpe for bpe in dp_sub_tokens[cur_mid]]
                dp_sub_tokens[end].append("##" + "".join(chars[cur_mid + 1:end + 1]))
            else:
                dp_sub_tokens[end] = [full_substr]
    # print(dp_sub_tokens)
    if len(chars) - 1 not in dp_sub_tokens:
        return ["[UNK]"]
    return dp_sub_tokens[len(chars) - 1]


class RandomWordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            # print("Token: ", token)
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                ends = list(range(start + 1, len(chars) + 1))
                random.shuffle(ends)
                # print(ends)
                # end = len(chars)
                cur_substr = None
                cur_end = start + 1
                for end in ends:
                    # while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        # print("curstr: ", cur_substr)
                        cur_end = end
                        # print("curend: ", cur_end)
                        break
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = cur_end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class ReverseWordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            end = len(chars)
            sub_tokens = []
            while end > 0:
                start = 0
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    start += 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                end = start

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(reversed(sub_tokens))
        return output_tokens


class bpeOTTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self,
                 vocab,
                 unk_token,
                 tag2idx,
                 full2part,
                 include_bi=True,
                 include_o=False,
                 M=None,
                 max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.M = M
        self.tag2idx = tag2idx
        self.include_bi = include_bi
        self.include_o = include_o
        self.full2part = full2part
        if isinstance(self.tag2idx, tuple):
            self.tag2idx = self.tag2idx[0]

    def tokenize(self, text, label):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        # print(label)
        # print(text)
        if self.include_bi == False and label != "O":
            label = label.split("-")[-1]
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            if token in self.vocab or label not in self.tag2idx:
                sub_tokens = tokenize_plain(chars, self.vocab)
            else:
                sub_tokens = tokenize_with_dp(chars, label, self.tag2idx, self.M, self.vocab, self.full2part)

            # print(sub_tokens)

            if sub_tokens == ["[UNK]"]:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class bpeOTTokenizer_global(object):
    """Runs WordPiece tokenization."""

    def __init__(self,
                 vocab,
                 unk_token,
                 tag2idx,
                 full2part,
                 word_part2idx,
                 idx2label_part,
                 word_idxes,
                 include_bi=True,
                 include_o=False,
                 M=None,
                 max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.M = M
        self.tag2idx = tag2idx
        self.include_bi = include_bi
        self.include_o = include_o
        self.full2part = full2part
        self.word_part2idx = word_part2idx
        self.idx2label_part = idx2label_part
        self.word_idxes = word_idxes
        if isinstance(self.tag2idx, tuple):
            self.tag2idx = self.tag2idx[0]

    def tokenize(self, text, label):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        # print(label)
        # print(text)
        if self.include_bi == False and label != "O":
            label = label.split("-")[-1]
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            if token in self.vocab or label not in self.tag2idx:
                sub_tokens = tokenize_plain(chars, self.vocab)
            else:
                sub_tokens = tokenize_with_dp_1(
                    chars,
                    label,
                    self.word_part2idx,
                    self.idx2label_part,
                    self.word_idxes,
                    self.M,
                    self.vocab,
                    self.full2part
                )

            # print(sub_tokens)

            if sub_tokens == ["[UNK]"]:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class OTTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self,
                 vocab,
                 unk_token,
                 word2subtokenlist,
                 word2subtokenlist_ratio,
                 max_input_chars_per_word=100):

        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.word2subtokenlist = word2subtokenlist
        self.word2subtokenlist_ratio = word2subtokenlist_ratio

    def tokenize(self, text, label):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        # print(self.word2subtokenlist_ratio)
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_normal = False
            label = label[2:]
            if token in self.word2subtokenlist_ratio and label in self.word2subtokenlist_ratio[token]:
                # print("get: ", token)

                ratios = np.array(self.word2subtokenlist_ratio[token][label])
                # lengths = np.array([len(i) for i in self.word2subtokenlist[token]])
                # ratios = ratios/lengths
                ratios = ratios / ratios.sum()
                if abs(ratios.sum() - 0) > 1e-10:
                    token_list = self.word2subtokenlist[token]
                    idx = np.random.choice(len(self.word2subtokenlist[token]), p=ratios)
                    sub_tokens = list(reversed(token_list[idx]))
                    output_tokens.extend(sub_tokens)
                else:
                    # print("stream1")
                    is_normal = True
            else:
                # print("stream2")
                is_normal = True
            # print("is normal: ", is_normal)
            if is_normal:
                is_bad = False
                start = 0
                sub_tokens = []
                while start < len(chars):
                    end = len(chars)
                    cur_substr = None
                    while start < end:
                        substr = "".join(chars[start:end])
                        if start > 0:
                            substr = "##" + substr
                        if substr in self.vocab:
                            cur_substr = substr
                            break
                        end -= 1
                    if cur_substr is None:
                        is_bad = True
                        break
                    sub_tokens.append(cur_substr)
                    start = end

                if is_bad:
                    output_tokens.append(self.unk_token)
                else:
                    output_tokens.extend(sub_tokens)
        return output_tokens


class MyBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, word_tokenizer="ot",
                 word2subtokenlist=None, word2subtokenlist_ratio=None, **kwargs):
        super(MyBertTokenizer, self).__init__(vocab_file=vocab_file,
                                              do_lower_case=do_lower_case,
                                              do_basic_tokenize=do_basic_tokenize,
                                              never_split=never_split,
                                              unk_token=unk_token,
                                              sep_token=sep_token,
                                              pad_token=pad_token,
                                              cls_token=cls_token,
                                              mask_token=mask_token,
                                              tokenize_chinese_chars=tokenize_chinese_chars,
                                              **kwargs)
        self.mode = word_tokenizer
        self.word2subtokenlist = word2subtokenlist
        self.word2subtokenlist_ratio = word2subtokenlist_ratio

        if not word_tokenizer == "":
            if word_tokenizer.startswith('ran'):
                self.wordpiece_tokenizer = RandomWordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            elif word_tokenizer.startswith('rev'):
                self.wordpiece_tokenizer = ReverseWordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            elif word_tokenizer == 'ot':
                self.wordpiece_tokenizer = OTTokenizer(
                    vocab=self.vocab,
                    unk_token=self.unk_token,
                    word2subtokenlist=self.word2subtokenlist,
                    word2subtokenlist_ratio=self.word2subtokenlist_ratio
                )

    def set_split_ratio(self, word2subtokenlist, word2subtokenlist_ratio):
        self.word2subtokenlist = word2subtokenlist
        self.word2subtokenlist_ratio = word2subtokenlist_ratio
        # self.change_tokenizer(self.mode)

    def load_split_ratio(self, data_dir=""):
        with open(data_dir + "ot_list.json", "r") as f:
            self.word2subtokenlist = json.load(f)
        with open(data_dir + "ot_ratio.json", "r") as f:
            self.word2subtokenlist_ratio = json.load(f)
        self.wordpiece_tokenizer = OTTokenizer(
            vocab=self.vocab,
            unk_token=self.unk_token,
            word2subtokenlist=self.word2subtokenlist,
            word2subtokenlist_ratio=self.word2subtokenlist_ratio
        )

    def dump_split_ratio(self, data_dir=""):
        import os
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        with open(data_dir + "ot_list.json", "w") as f:
            json.dump(self.word2subtokenlist, f)
        with open(data_dir + "ot_ratio.json", "w") as f:
            json.dump(self.word2subtokenlist_ratio, f)

    def ot_tokenize(self, text, label):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token, label):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text, label)
        return split_tokens


if __name__ == '__main__':
    tokenizer = MyBertTokenizer(vocab_file="vocab.txt")
