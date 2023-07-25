import pandas as pd
import numpy as np
import csv
from dataset.util import *
import torch
import tqdm

def get_nameAndId(categories_path):
    df = pd.read_csv(categories_path).values
    name_to_id = {}
    id_to_name = {}

    for item in df:
        name_to_id[item[1]] = item[0]
        id_to_name[item[0]] = item[1]

    return name_to_id, id_to_name

if __name__ == '__main__':
    categories_path = '../dataset/data/category.csv'
    embed_file_path = '../embeddings/SD-CEM#JP#50.csv'
    checkIn_path = '../dataset/data/CheckinLocationCategoryIDSequenceJP50Filter.csv'

    name_to_id, id_to_name = get_nameAndId(categories_path)

    embeddings = {}
    with open(embed_file_path) as f:
        for row in csv.reader(f):
            embeddings[name_to_id[row[0]]] = torch.Tensor(list(map(float, row[1:])))
    embed_size = len(list(embeddings.values())[0])

    ai_2_bi = {}
    bi_2_ai = {}
    index = 0
    for item in embeddings.keys():
        ai_2_bi[index] = item
        bi_2_ai[item] = index
        index += 1

    embeds = torch.zeros((len(embeddings), embed_size))
    for i, item in enumerate(embeddings.values()):
        embeds[i] = item

    corpus = []
    with open(checkIn_path, encoding='utf-8') as file:
        for row in csv.reader(file):
            temp = []
            for item in row:
                venue, category = item.split("@")
                temp.append(category)
            corpus.append(temp)
    file.close()

    mrrs = []
    accs = []
    for line in tqdm.tqdm(corpus):
        test_x = line[:-1]
        test_y = line[1:]

        predict_list_mrr = []
        predict_list_acc = []

        pred_embeddings = torch.zeros((len(test_x), embed_size))
        for i, item in enumerate(test_x):
            pred_embeddings[i] = embeddings[item]

        pred = torch.matmul(pred_embeddings, embeds.T)
        p = torch.nn.functional.softmax(torch.Tensor(pred), dim=-1)

        predict = torch.argsort(p, dim=-1, descending=True)

        predict_list_acc += list(predict.squeeze(dim=0)[:, :5])
        predict_list_mrr += list(predict.squeeze(dim=0))

        counts = 0
        for i in range(len(predict_list_acc)):
            if bi_2_ai[test_y[i]] in predict_list_acc[i]:
                counts += 1
        accuracy = counts / len(predict_list_acc)
        accs.append(accuracy)

        score = 0
        mrr = 0
        for i in range(len(predict_list_mrr)):
            l = predict_list_mrr[i]
            truth = bi_2_ai[test_y[i]]
            for j in range(len(l)):
                if truth == l[j]:
                    rank = 1 / (j + 1)
                    break
            score += rank
        mrr = score / len(test_y)
        mrrs.append(mrr)

    print(f"accuracy: {np.mean(accs)} mrr: {np.mean(mrrs)}")