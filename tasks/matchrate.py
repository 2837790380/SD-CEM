# %load eval.py
from dataset.util import *
import torch
from model import Model, SDCEM
import csv


def ancient_dict(categories_path):
    df = pd.read_csv(categories_path, header=None)
    df = df.values

    venue_size = len(df)
    parent_list = {}
    for i in range(1, venue_size):
        if df[i][2] == 'root':
            parent_list[df[i][0]] = df[i][0]
            continue
        parent = df[i][2][-5 - 24: -5]
        parent_list[df[i][0]] = parent

    return parent_list


def most_similar(query, query_embedding, all_embeddings, top=5):

    vocab_size = len(all_embeddings)
    sim = np.zeros(vocab_size)
    for i, item in enumerate(all_embeddings.values()):
        sim[i] = torch.nn.functional.cosine_similarity(
            torch.Tensor(item), torch.Tensor(query_embedding), dim=0)

    count = 0
    ms_list = []
    for i in (-1 * sim).argsort():
        if list(all_embeddings.keys())[i] == query:
            continue

        count += 1
        ms_list.append(list(all_embeddings.keys())[i])
        if count >= top:
            return ms_list

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
    embed_file_path = '../embeddings/SD-CEM#US#50.csv'
    name_to_id, id_to_name = get_nameAndId(categories_path)
    parent_dict = ancient_dict(categories_path)

    embeddings = {}
    with open(embed_file_path) as f:
        for row in csv.reader(f):
            embeddings[name_to_id[row[0]]] = torch.Tensor(list(map(float, row[1:])))

    match_counts = 0
    for key, value in embeddings.items():
        l = most_similar(key, value, embeddings, 5)[0]
        if parent_dict[key] == parent_dict[l]:
            match_counts += 1

    print(f" match_counts is %d, sum is %d, rate is %f" %
          (match_counts, len(embeddings), match_counts / len(embeddings)))
