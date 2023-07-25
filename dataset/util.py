import sys
import os
import numpy as np
import pandas as pd
import csv
import collections
import copy
import random


def preprocessData(train_data, path_map, len_seq, len_level, is_train):
    path_map_order = getWrongPathOrder(path_map, len_level)

    T_encoder_inputs = path_map[train_data]
    F_encoder_inputs = path_map_order[train_data]
    decoder_inputs = np.zeros_like(train_data)

    if is_train:
        for i in range(len(train_data)):
            seq = train_data[i]
            seq_len = len_seq[i]
            T_encoder_input = path_map[seq]
            F_encoder_input = path_map_order[seq]
            mask_idx = []

            for idx in range(seq_len):
                if random.random() < 0.15:
                    # 80%的概率替换为mask标记
                    mask_idx.append(idx)
                    if random.random() < 0.8:
                        path = path_map[seq[idx]].copy()
                        path_F = path_map_order[seq[idx]].copy()
                        try:
                            lenOfPath = len_level[seq[idx]-1]+1
                        except:
                            print(seq)
                            print(seq[idx])
                        q = np.array(range(lenOfPath), dtype=np.int16)
                        P_select_mask = np.exp(q-lenOfPath) / np.sum(np.exp(q-lenOfPath))
                        mask_path_idx = np.random.choice(q, size=1, p=P_select_mask)
                        path[mask_path_idx] = 0
                        path_F[mask_path_idx] = 0

                        T_encoder_input[idx] = path
                        F_encoder_input[idx] = path_F

                    else:
                        # 10%的概率变为词料库中随机的单词
                        if random.random() < 0.5:
                            random_venue_idx = np.random.choice(range(len(len_level)), size=1)+1
                            seq[idx] = random_venue_idx
                            path = path_map[seq[idx]].copy()
                            path_F = path_map_order[seq[idx]].copy()

                            T_encoder_input[idx] = path
                            F_encoder_input[idx] = path_F
                        # 10%的概率不变
                        else:
                            continue
            T_encoder_inputs[i] = T_encoder_input
            F_encoder_inputs[i] = F_encoder_input

            decoder_input = np.zeros_like(seq)
            decoder_input[mask_idx] = seq[mask_idx]
            decoder_inputs[i] = decoder_input

        T_flags = np.ones((len(train_data), 1))
        F_flags = np.zeros_like(T_flags)

        encoder_inputs = np.concatenate([T_encoder_inputs, F_encoder_inputs], axis=0)
        decoder_inputs_ = np.concatenate([decoder_inputs, decoder_inputs], axis=0)
        path_order_flags = np.concatenate([T_flags, F_flags], axis=0)
    else:
        encoder_inputs = T_encoder_inputs
        decoder_inputs_ = train_data.copy()
        path_order_flags = np.ones((len(train_data), 1))
        
    return encoder_inputs, decoder_inputs_, path_order_flags


def getWrongPathOrder(path_map, len_level):
    path_map_order = path_map.copy()
    for i, path in enumerate(path_map[1:-1]):
        l = len_level[i]
        position = random.sample(range(0, l+1), 2)
        temp = path[position[0]]
        path[position[0]] = path[position[1]]
        path[position[1]] = temp
        path_map_order[i+1] = path
    
    return path_map_order


def get_cia_map(categories_path, venue_to_id):
    # 返回一个array，索引是venue的id，对应的value是其一系列父节点
    cia_map = np.zeros(len(venue_to_id), dtype=object)
    df = pd.read_csv(categories_path).values
    for i in range(len(df)):
        cia_id_list = []
        ci = venue_to_id[df[i][0]]
        cia_list = df[i][2].split('$')
        for cia in cia_list:
            cia_id_list += [venue_to_id[cia]]
        cia_map[ci] = np.array(cia_id_list)

    return cia_map


def get_pathAndLevel(categories_path, venue_to_id):
    level_num = 6
    df = pd.read_csv(categories_path).values
    path_map = np.zeros((len(venue_to_id), level_num), dtype=np.int16)
    level_map = np.repeat(np.arange(level_num).reshape(1, -1), len(venue_to_id), axis=0)
    len_level = []
    for i in range(len(df)):
        path_id_list = []

        ci = venue_to_id[df[i][0]]
        path_list = df[i][2].split('$')
        path_id_list += [ci]  # path中包含ei
        for cia in path_list[:-1]:
            path_id_list += [venue_to_id[cia]]

        path_id_list.reverse()

        level_map[ci, len(path_id_list):] = level_num
        len_level.append(len(path_id_list) - 1)

        path_map[ci][:len(path_id_list)] = np.array(path_id_list)

    return path_map, level_map, np.array(len_level)


def extension(cia_map, xs):
    temp = []
    sum_weight = 0
    
    cias = cia_map[xs]
    for cia in cias:
        temp += list(cia)
    na = collections.Counter(temp)
    weight_array = copy.deepcopy(cias)
    for i in range(len(cias)):
        for j in range(len(cias[i])):
            weight_array[i][j] = na[cias[i][j]]/(j+1)
            sum_weight += weight_array[i][j]
    return cias, weight_array/sum_weight


def preprocess_2d(categorys_path, sequence_path):
    specials = ['<pad>', '<eos>', '<sos>', '<mask>']
    labelOfCategory = {
        '4d4b7104d754a06370d81259': 1,
        '4d4b7105d754a06372d81259': 2,
        '4d4b7105d754a06373d81259': 3,
        '4d4b7105d754a06374d81259': 4,
        '4d4b7105d754a06376d81259': 5,
        '4d4b7105d754a06377d81259': 6,
        '4d4b7105d754a06375d81259': 7,
        '4e67e38e036454776db1fb3a': 8,
        '4d4b7105d754a06378d81259': 9,
        '4d4b7105d754a06379d81259': 10
    }
    df = pd.read_csv(categorys_path, header=None)
    df = df.values

    venue_size = len(df)
    parent_list = {}

    for i in range(1, venue_size):
        if df[i][2] == 'root':
            parent_list[df[i][0]] = df[i][0]
            continue
        parent = df[i][2][-5 - 24: -5]
        parent_list[df[i][0]] = parent

    ids = df[1:, 0]
    venue_to_id = {tok: i for i, tok in enumerate(specials)}
    id_to_venue = {i: tok for i, tok in enumerate(specials)}

    for id in ids:
        if id not in venue_to_id:
            new_idx = len(venue_to_id)
            venue_to_id[id] = new_idx
            id_to_venue[new_idx] = id

    corpus = []
    corpus_label = []
    with open(sequence_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            temp = list([venue_to_id[venue] for venue in row])
            temp_label = list([labelOfCategory[parent_list[venue]] for venue in row])
            corpus.append(temp)
            corpus_label.append(temp_label)

    return corpus, corpus_label, venue_to_id, id_to_venue, parent_list


def preprocess_mini_seq(corpus, corpus_label):
    mini_seq = []
    next_seq = []
    labelofseq = []
    for c, cl in zip(corpus, corpus_label):
        c_s = np.array(c)[:-1]
        c_d = np.array(c)[1:]

        cl = np.array(cl)[:-1]
        for i in range(1, 11):
            mask = cl == i
            if len(c_s[mask]) < 2:
                continue
            mini_seq.append(list(c_s[mask]))
            labelofseq.append(i)
            next_seq.append(list(c_d[mask]))
    return mini_seq, next_seq, labelofseq


def preprocess_mini_seq_2(corpus, corpus_label):
    mini_seq = []
    next_seq = []
    labelofseq = []
    for c, cl in zip(corpus, corpus_label):
        c_s = np.array(c)[:-1]
        c_d = np.array(c)[1:]

        cl = np.array(cl)[:-1]

        mini_seq.append(list(c_s))
        labelofseq.append(cl)
        next_seq.append(list(c_d))
    return mini_seq, next_seq, labelofseq


def creat_contexts_target(corpus, window_size=1):
    contexts = []
    target = []

    for temp in corpus:
        target += temp[window_size:-window_size]
        for idx in range(window_size, len(temp) - window_size):
            cs = []
            for t in range(-window_size, window_size + 1):
                if t == 0:
                    continue
                cs.append(temp[idx + t])
            contexts.append(cs)

    return np.array(contexts), np.array(target)


def get_mob(vocab_size, corpus):
    mob = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for corpu in corpus:
        for i in range(len(corpu[:-1])):
            mob[corpu[i], corpu[i + 1]] += 1
    mob = mob / mob.sum()
    # return torch.tensor(mob, requires_grad=False).float()
    return mob
