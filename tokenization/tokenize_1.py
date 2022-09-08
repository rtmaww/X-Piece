def search_part_score(M, label_id, full2part, fullid):
    if fullid in full2part:
        return M[label_id, full2part[fullid]]
    else:
        return 1e-9


def tokenize_with_dp_1(chars, label, word_part2idx, idx2label_part, word_idxes, M, vocab, full2part):
    dp_sub_tokens = {}
    dp_scores = {}
    start_mid = 0
    word = "".join(chars)
    word_id = word_part2idx[word]
    label_num = word_idxes[word_id]
    word_label_id = -1
    for label_id in range(word_id, word_id + label_num):
        if idx2label_part[label_id] == label:
            word_label_id = label_id
            break
    # search for the beginning subword for marginal condition.
    for i in range(len(chars)):
        substr = "".join(chars[0:i + 1])
        if substr in vocab:
            dp_sub_tokens[i] = [substr]
            dp_scores[i] = search_part_score(M, word_label_id, full2part, vocab[substr])
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
            cur_score = search_part_score(M, word_label_id, full2part, vocab[full_substr])
        else:
            cur_score = -1e9
        # print(cur_score)
        cur_mid = -1
        for mid in range(start_mid, end):
            if mid not in dp_scores:
                continue
            substr = "##" + "".join(chars[mid + 1:end + 1])
            if substr in vocab and dp_scores[mid] * search_part_score(M, word_label_id, full2part, vocab[substr]) > cur_score:
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
                cur_score = dp_scores[mid] * search_part_score(M, word_label_id, full2part, vocab[substr])
                cur_mid = mid
        if cur_score != -1e9:
            dp_scores[end] = cur_score
            if cur_mid != -1:
                dp_sub_tokens[end] = [bpe for bpe in dp_sub_tokens[cur_mid]]
                dp_sub_tokens[end].append("##" + "".join(chars[cur_mid + 1:end + 1]))
            else:
                dp_sub_tokens[end] = [full_substr]
    # print(dp_sub_tokens)
    if len(chars)-1 not in dp_sub_tokens:
        return ["[UNK]"]
    return dp_sub_tokens[len(chars)-1]