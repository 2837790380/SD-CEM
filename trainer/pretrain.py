import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import SDCEM, Model
from .optim_schedule import ScheduledOptim

import tqdm


class Trainer:
    """
    Trainer make the pretrained model with two training method.

        1. Task #1: Masked Category Model
        2. Task #2: Category transition Prediction
    """

    def __init__(self, model: Model, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 max_epoch=40, with_cuda: bool = True, cuda_device=5, log_freq: int = 10):
        """
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(f'cuda:{cuda_device}' if cuda_condition else "cpu")

        # This model will be saved every epoch
        self.model = model

        self.sdcem = SDCEM(self.model, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.sdcem.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.hidden, n_warmup_steps=warmup_steps)

        self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion2 = nn.NLLLoss(ignore_index=0, reduction='sum')

        self.log_freq = log_freq
        self.preTrain_epochs = int(0.5 * max_epoch)

        print("Total Parameters:", sum([p.nelement() for p in self.sdcem.parameters()]))

    def train(self, epoch, alpha):
        self.iteration(epoch, self.train_data, alpha)

    def test(self, epoch, alpha):
        self.iteration(epoch, self.test_data, alpha, train=False)

    def iteration(self, epoch, data_loader, alpha, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_next_loss = 0.0
        avg_mcm_loss = 0.0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            mask_lm_output, next_tm_output = self.sdcem.forward(data["input"], data['source_input'], data['next_input'],
                                                                data['length'])

            next_loss = self.criterion2(next_tm_output.transpose(1, 2), data['next_input'])

            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["label"])

            if epoch < self.preTrain_epochs:
                loss = mask_loss
                avg_loss += loss.item()
            else:
                loss = alpha * mask_loss + (1 - alpha) * next_loss
                avg_loss += loss.item()

                avg_mcm_loss += mask_loss.item()
                avg_next_loss += next_loss.item()

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

        if epoch < self.preTrain_epochs:
            print("EP%d_%s, avg_loss=" % (epoch, str_code), (avg_loss / len(data_iter)))
        else:
            print("EP%d_%s, avg_loss=" % (epoch, str_code), (avg_loss / len(data_iter)))
            print("EP%d_%s, avg_mcm_loss=" % (epoch, str_code), (avg_mcm_loss / len(data_iter)))
            print("EP%d_%s, avg_next_loss=" % (epoch, str_code), (avg_next_loss / len(data_iter)))

    def save(self, epoch, file_path="output/trained.model"):
        """
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
