from trainer import Trainer
from dataset import mydataset
from dataset.util import *
from torch.utils.data import DataLoader
from model import Model
import torch
import argparse


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    # setup_seed(1)

    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--city', '-c', type=str, default='JP')
    parser.add_argument('--train_size', '-ts', type=float, default=0.8)
    parser.add_argument('--embedding_size', '-es', type=int, default=20)
    parser.add_argument('--ts_ratio', '-tsr', type=float, default=1)
    parser.add_argument('--cuda_device', '-cd', type=int, default=0)
    parser.add_argument('--epochs', '-e', type=int, default=40)
    parser.add_argument('--alpha', '-a', type=float, default=1.0)
    parser.add_argument('--seed', '-s', type=int, default=1)

    args = parser.parse_args()
    print(args)

    setup_seed(args.seed)

    categories_path = './dataset/data/category.csv'
    sequence_path = f'./dataset/data/CheckinCategoryIDSequence{args.city}.csv'
    corpus, corpus_label, venue_to_id, id_to_venue, parent_list = preprocess_2d(categories_path, sequence_path)
    path_map, level_map, len_level = get_pathAndLevel(categories_path, venue_to_id)

    train_corpus = corpus
    train_corpus_label = corpus_label
    mob = get_mob(len(venue_to_id), train_corpus)
    mini_seqs, next_seqs, labelofseqs = preprocess_mini_seq(train_corpus, train_corpus_label)

    embedding_dim_list = [args.embedding_size]

    train_size = int(args.ts_ratio * len(mini_seqs))

    train_data = mini_seqs[:train_size]
    train_next_data = next_seqs[:train_size]
    train_labels = labelofseqs[:train_size]

    test_data = mini_seqs[train_size:]
    test_next_data = next_seqs[train_size:]
    test_labels = labelofseqs[train_size:]
    withTest = False

    train_dataset = mydataset.MyDataset(train_data, train_next_data, labelofseqs, len(venue_to_id), 100)  # mini_seq:len is 100, origin:len is 300
    test_dataset = mydataset.MyDataset(test_data, test_next_data, labelofseqs, len(venue_to_id), 100)

    train_data_loader = DataLoader(train_dataset, batch_size=128, num_workers=0, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=128, num_workers=0) \
        if withTest is True else None

    mob_1 = mob.copy()
    model = Model(torch.tensor(mob_1, requires_grad=False).float(), path_map, len_level, len(venue_to_id), embedding_dim=args.embedding_size,
                hidden_dim=128, n_layers=8, attn_heads=8, dropout=0.1)

    trainer = Trainer(model, len(venue_to_id), train_dataloader=train_data_loader,
                          test_dataloader=test_data_loader, lr=1e-3,
                          betas=(0.9, 0.999),
                          weight_decay=0.0,
                          with_cuda=True, cuda_device=args.cuda_device,
                          log_freq=20, warmup_steps=1000, max_epoch=args.epochs)

    for epoch in range(args.epochs):
        trainer.train(epoch, args.alpha)
        if (epoch+1) % 10 == 0:
            trainer.save(epoch, f'./output/seed_{args.seed}_{args.alpha}_{args.embedding_size}_{args.ts_ratio}_{args.city}.model')

