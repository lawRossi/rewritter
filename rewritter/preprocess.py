from collections import defaultdict
import numpy as np
from itertools import chain
import json
import os
import re
from itertools import permutations


def longest_common_sequence(sequence1, sequence2):
    dp = [[0] * (len(sequence2) + 1) for _ in range(len(sequence1) + 1)]
    for i in range(1, len(sequence1) + 1):
        for j in range(1, len(sequence2) + 1):
            if sequence1[i-1] == sequence2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    i = len(sequence1)
    j = len(sequence2)
    common_sequence = []
    while i > 0 and j > 0:
        if sequence1[i-1] == sequence2[j-1]:
            common_sequence.append(sequence1[i-1])
            i -= 1
            j -= 1
        elif dp[i][j-1] > dp[i-1][j]:
            j -= 1
        else:
            i -= 1
    common_sequence = common_sequence[::-1]
    if isinstance(sequence1, str):
        common_sequence = "".join(common_sequence)
    return common_sequence


def longest_matched_prefex(sequence1, sequence2):
    dp = [[0] * (len(sequence2) + 1) for _ in range(len(sequence1) + 1)]
    max_len = 0
    sub_string = []
    for i in range(1, len(sequence1) + 1):
        for j in range(1, len(sequence2) + 1):
            if sequence1[i-1] == sequence2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len and j - dp[i][j] == 0:
                    max_len = dp[i][j]
                    sub_string = sequence2[j-max_len:j]
    return sub_string


def find(text , query, pos=0):
    i = pos
    while i <= len(text) - len(query):
        for j in range(len(query)):
            if text[i+j] != query[j]:
                break
        else:
            return i
        i += 1
    return -1


def rfind(text, query):
    i = len(text) - len(query)
    while i >= 0:
        for j in range(len(query)):
            if text[i+j] != query[j]:
                break
        else:
            return i
        i -= 1
    return -1


def label_history_span(history, span, labels, label):
    idx = rfind(history, span)
    if idx != -1:
        for j in range(idx, idx + len(span)):
            if labels[j] != 'O':
                if isinstance(labels[j], list):
                    labels[j].append(label)
                else:
                    labels[j] = [labels[j], label]
            else:
                labels[j] = label
    else:
        prefix = longest_matched_prefex(history, span)
        if prefix == span or not prefix:
            return
        postfix = span[len(prefix):]
        label_history_span(history, prefix, labels, label)
        label_history_span(history, postfix, labels, label)


def label_history_spans(history, spans, labels, label):
    for i, span in enumerate(spans):
        label_ = f"{label}{i+1}"
        label_history_span(history, span, labels, label_)


def auto_label(history, source, target):
    replacer_groups, replaced_groups, inserted_groups = get_semantic_role_groups(source, target)
    history_labels = ['O'] * len(history)
    source_labels = ['O'] * (len(source) + 1)
    for i, group in enumerate(replaced_groups):
        for j in group:
            source_labels[j] = f"DELETE{i+1}"
    replacers = [target[group[0]:group[-1]+1] for group in replacer_groups]
    label_history_spans(history, replacers, history_labels, "EXTRACT")
    inserts = [target[group[0]:group[-1]+1] for group in inserted_groups]
    label_history_spans(history, inserts, history_labels, "COPY")
    replaced, offsets = replace(source, target, replaced_groups, replacer_groups)
    inserted_len = 0
    for i, (group, insert) in enumerate(zip(inserted_groups, inserts)):
        e = group[0]
        if e == 0:
            source_labels[0] = f"INSERT{i+1}"
            replaced = insert + replaced
        else:
            prefix = target[:e]
            idx = find(replaced, prefix) + len(prefix)
            replaced = replaced[:idx] + insert + replaced[idx:]
            if idx != -1:
                idx -= inserted_len
                if idx >= len(offsets):
                    idx = -1
                else:
                    idx += offsets[idx]
            source_labels[idx] = f"INSERT{i+1}"
        inserted_len += len(insert)
    return history_labels, source_labels


def get_semantic_role_groups(source, target):
    replaced = []
    inserted = []
    common_sequence = longest_common_sequence(source, target)
    replaced = list(range(len(source)))
    offset = 0
    for i, c in enumerate(source):
        idx = find(common_sequence, [c], offset)
        if idx != -1 and i >= idx:
            replaced.remove(i)
            offset = idx + 1
  
    inserted = list(range(len(target)))
    offset = 0
    for i, c in enumerate(common_sequence):
        idx = find(target, [c], offset)
        if idx != -1 and len(target) - idx >= len(common_sequence) - i:
            inserted.remove(idx)
            offset = idx + 1
    replaced_groups = group(replaced)
    inserted_groups = group(inserted)
    replacer = []
    inserted_len = 0
    shortened = 0
    for i, inserted_group in enumerate(inserted_groups):
        for replaced_group in replaced_groups:
            if inserted_group[0] - inserted_len == replaced_group[0] - shortened:
                replacer.append(i)
                shortened += len(replaced_group) - len(inserted_group)
                break
        else:
            inserted_len += len(inserted_group)
    replacer_groups = [inserted_groups[i] for i in replacer]
    inserted_groups = [inserted_groups[i] for i in range(len(inserted_groups)) if i not in replacer]
    return replacer_groups, replaced_groups, inserted_groups


def group(elements):
    groups = []
    g = []
    for i in range(len(elements)):
        g.append(elements[i])
        if i == len(elements) - 1 or elements[i] != elements[i+1] - 1:
            groups.append(g)
            g = []
    return groups


def replace(source, target, replaced_groups, replacer_groups):
    i = 0
    result = []
    offsets = []
    offset = 0
    for replaced, replacer in zip(replaced_groups, replacer_groups):
        while i < replaced[0]:
            result.append(source[i])
            offsets.append(offset)
            i += 1
        i += len(replaced)
        offset += len(replaced) - len(replacer)
        for j in replacer:
            result.append(target[j])
            offsets.append(offset)
    while i < len(source):
        result.append(source[i])
        offsets.append(offset)
        i += 1
    if isinstance(source, str):
        result = "".join(result)
    return result, offsets


mapping_labels = {
    "EXTRACT": "DELETE",
    "COPY": "INSERT",
}

label_mapping = {
    "EXTRACT": 1,
    "COPY": 2
}


def make_edit_matrix(history_labels, source_labels):
    matrix = [[0] * len(source_labels) for _ in range(len(history_labels))]
    for i, label in enumerate(history_labels):
        if label != "O":
            if not isinstance(label, list):
                labels = [label]
            else:
                labels = label
            for label in labels:
                mapping_label = mapping_labels[label[:-1]] + label[-1]
                for j in range(len(source_labels)):
                    if source_labels[j] == mapping_label:
                        matrix[i][j] = label_mapping[label[:-1]]
    return matrix


def get_operations(history, edit_matrix):
    processed = np.zeros_like(edit_matrix)
    operations = []
    edit_matrix = np.array(edit_matrix)
    for i in range(edit_matrix.shape[0]):
        for j in range(edit_matrix.shape[1]):
            if edit_matrix[i][j] != 0 and processed[i][j] == 0:
                if edit_matrix[i][j] == 2: # 插入
                    token_idxes = []
                    for r in range(i, edit_matrix.shape[0]):
                        if edit_matrix[r][j] == 2:
                            token_idxes.append(r)
                    tokens_list = token_idxes_2_tokens(history, token_idxes)
                    operations.append(("insert", j, tokens_list))
                    processed[:, j] = 1
                else:  # 替换
                    replaced_pos = []
                    for c in range(j, edit_matrix.shape[1]):
                        if edit_matrix[i][c] == 1:
                            replaced_pos.append(c)
                        else:
                            break
                    token_idxes = []
                    max_r = i
                    for r in range(i, edit_matrix.shape[0]):
                        if edit_matrix[r][j] == 1:
                            token_idxes.append(r)
                            max_r = r
                    tokens_list = token_idxes_2_tokens(history, token_idxes)
                    operations.append(("replace", replaced_pos, tokens_list))
                    processed[i:max_r+1,j:max(replaced_pos)+1] = 1
    return operations


def token_idxes_2_tokens(context, token_idxes):
    groups = group(token_idxes)
    tokens_list = []
    for g in groups:
        tokens_list.append([context[i] for i in g])
    return tokens_list


def translate(source, operations):
    results = [source.copy()]
    operations = sorted(operations, key=lambda x: x[1][0] if isinstance(x[1], list) else x[1])
    offset = 0
    for operation in operations:
        pos, tokens_list = operation[1], operation[2]
        if operation[0] == "insert":
            new_results = []
            for tokens in permute_tokens(tokens_list):
                for result in results:
                    result = result[:pos+offset] + tokens + result[pos+offset:]
                    new_results.append(result)
            offset += len(tokens)
            results = new_results
        else:
            new_results = []
            for tokens in permute_tokens(tokens_list):
                for result in results:
                    result = result[:pos[0]+offset] + tokens + result[pos[-1]+offset+1:]
                    new_results.append(result)
            offset += len(tokens) - len(pos)
            results = new_results
    return results


def permute_tokens(tokens_list):
    for permutation in permutations(tokens_list):
        yield list(chain.from_iterable(permutation))


def convert_dataset(data_file, save_dir, tokenize, min_tf=1, max_ctx_len=80, 
        max_cur_len=30, sep="\t\t", bert=False, test_size=0.1, appendix=""):
    with open(data_file, encoding="utf-8") as fi:
        data = []
        for line in fi:
            splits = line.strip().split(sep)
            data.append([re.sub("\s+", " ", split.lower()) for split in splits])

    tokenized_data, vocab = tokenize_and_build_vocabulary(data, tokenize, min_tf, appendix)

    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as fo:
        json.dump(vocab, fo)

    num_train = int(len(tokenized_data) * (1 - test_size))

    n = 0
    with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as fo:
        for sample, tokenized_sample in zip(data[:num_train], tokenized_data[:num_train]):
            result = convert_sample(sample, tokenized_sample, vocab, max_ctx_len, max_cur_len)
            if result:
                n += 1
                fo.write(json.dumps(result) + "\n")
    print(n)
    n = 0
    with open(os.path.join(save_dir, "test.json"), "w", encoding="utf-8") as fo:
        for sample, tokenized_sample in zip(data[num_train:], tokenized_data[num_train:]):
            result = convert_sample(sample, tokenized_sample, vocab, max_ctx_len, max_cur_len)
            if result:
                n += 1
                fo.write(json.dumps(result) + "\n")
    print(n)


def tokenize_and_build_vocabulary(data, tokenize, min_tf, appendix):
    tokenized_data = []
    counts = defaultdict(int)
    for sample in data:
        tokenized_sample = []
        for text in sample:
            tokens = tokenize(text)
            tokens.append("<SEP>")
            for token in tokens:
                counts[token] += 1
            tokenized_sample.append(tokens)
        # add appendix to context
        tokenized_sample[-3].extend(appendix.split(" "))
        tokenized_data.append(tokenized_sample)
    words = {word for word, count in counts.items() if count >= min_tf}
    words.add("<UNK>")
    vocab = {word: i + 1 for i, word in enumerate(words)}
    return tokenized_data, vocab


def convert_sample(sample, tokenized_sample,  vocab, max_ctx_len, max_cur_len):
    context = list(chain.from_iterable(sample[:2]))
    context = [vocab.get(token, vocab["<UNK>"]) for token in context]
    context = context[-max_ctx_len:]
    utterance = tokenized_sample[-2]
    utterance = [vocab.get(token, vocab["<UNK>"]) for token in utterance]
    utterance = utterance[-max_cur_len:]
    reference = tokenized_sample[-1]
    reference = [vocab.get(token, vocab["<UNK>"]) for token in reference]
    context_labels, utterance_labels = auto_label(context, utterance[:-1], reference[:-1])
    matrix = make_edit_matrix(context_labels, utterance_labels)
    operations = get_operations(context, matrix)
    targets = translate(utterance, operations)
    if reference != targets[0]:
        return {}
    matrix = matrix + [[-1] * len(utterance) for _ in range(max_ctx_len - len(context))]
    for i in range(len(matrix)):
        matrix[i] += [-1] * (max_cur_len - len(utterance))
    context += [0] * (max_ctx_len - len(context))
    utterance += [0] * (max_cur_len - len(utterance))
    labels = list(chain.from_iterable(matrix))
    return {"context": context, "utterance": utterance, "labels": labels, "raw_sample": sample}


if __name__ == "__main__":
    assert longest_common_sequence("abde", "acdef") == "ade"
    assert longest_common_sequence("abde", "acfde") == "ade"
    assert longest_common_sequence("abde", "acdefg") == "ade"
    assert longest_common_sequence("abde", "acdxxefg") == "ade"
    assert longest_matched_prefex("abce", "bcd") == "bc"
    assert longest_matched_prefex("abce", "bcde") == "bc"
    assert longest_matched_prefex("abce", "abbcde") == "ab"
    assert longest_matched_prefex(["a", "b", "c", "e"], ["b","c","d","e"]) == ["b", "c"]

    # ## 实例
    # history_labels, source_labels = auto_label(list("梅西和C罗你喜欢谁"), list("他们我都喜欢"), list("梅西C罗我都喜欢"))
    # assert history_labels == ["EXTRACT1", "EXTRACT1", "O", "EXTRACT1", "EXTRACT1", "O", "O", "O", "O"]
    # assert source_labels == ["DELETE1", "DELETE1", "O", "O", "O", "O", "O"]
    # matrix = make_edit_matrix(history_labels, source_labels)
    # print(matrix)
    # operations = get_operations(list("梅西和C罗你喜欢谁"), matrix)
    # print(operations)
    # print(translate(list("他们我都喜欢"), operations))

    # history_labels, source_labels = auto_label(list("梅西和C罗你喜欢谁|梅西"), list("为什么"), list("为什么喜欢梅西"))
    # assert history_labels == ["O", "O", "O", "O", "O", "O", "COPY1", "COPY1", "O", "O", "COPY1", "COPY1"]
    # assert source_labels == ["O", "O", "O", "INSERT1"]
    # matrix = make_edit_matrix(history_labels, source_labels)
    # print(matrix)
    # operations = get_operations(list("梅西和C罗你喜欢谁|梅西"), matrix)
    # print(operations)
    # print(translate(list("为什么"), operations))

    # history_labels, source_labels = auto_label(list("小王真烦|我想打他"), list("他？"), list("想打小王？"))
    # assert history_labels == ["EXTRACT1", "EXTRACT1", "O", "O", "O", "O", "EXTRACT1", "EXTRACT1", "O"]
    # assert source_labels == ["DELETE1", "O", "O"]
    # matrix = make_edit_matrix(history_labels, source_labels)
    # print(matrix)
    # print(get_operations(list("小王真烦|我想打他"), matrix))

    context = "今天的股市大盘行情怎样 股票今天涨了吗"
    source = "没涨"
    target = " 今天股票没涨"
    context_labels, utter_labels = auto_label(context, source, target)
    print(context_labels)
    print(utter_labels)
