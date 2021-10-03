import os
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from glob import glob
from tqdm import trange

from utils.logwriter import Logwriter
from utils.utils import load_npy_data
import utils.pytorch_util as ptu

from model import Discriminator, Generator


class CycleGANTrainer:

    def __init__(self, args):
        self.args = args
        self.logwriter = Logwriter(args.directory)
        self.train_stats_writer = self.logwriter.get_page_writer("train_stats")
        self.test_stats_writer = self.logwriter.get_page_writer("test_stats")
        self.t = 0
        self.init_model()


    def init_model(self):
        # models
        self.discriminatorA = Discriminator().to('cuda' if ptu._use_gpu else 'cpu')
        self.discriminatorB = Discriminator().to('cuda' if ptu._use_gpu else 'cpu')
        self.generatorAB = Generator().to('cuda' if ptu._use_gpu else 'cpu')
        self.generatorBA = Generator().to('cuda' if ptu._use_gpu else 'cpu')
        # loss
        self.abs_criterion = nn.L1Loss()
        self.criterionGAN = nn.MSELoss()
        # data
        self.dataA = glob('datasets/{}/train/*.*'.format(self.args.dataset_A_dir))
        self.dataB = glob('datasets/{}/train/*.*'.format(self.args.dataset_B_dir))
        self.batch_idxs = min(min(len(self.dataA), len(self.dataB)), self.args.train_size) // self.args.batch_size
        self.dataA_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_A_dir))
        self.dataB_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_B_dir))
        self.batch_idxs_test = min(min(len(self.dataA_test), len(self.dataB_test)), self.args.train_size) // self.args.batch_size
        # optimizers
        self.discriminatorA_optimizer = torch.optim.Adam(self.discriminatorA.parameters(), lr=self.args.lr)
        self.discriminatorB_optimizer = torch.optim.Adam(self.discriminatorB.parameters(), lr=self.args.lr)
        self.generatorAB_optimizer = torch.optim.Adam(self.generatorAB.parameters(), lr=self.args.lr)
        self.generatorBA_optimizer = torch.optim.Adam(self.generatorBA.parameters(), lr=self.args.lr)


    def train(self):
        """ train one epoch """
        # shuffle training data
        np.random.shuffle(self.dataA)
        np.random.shuffle(self.dataB)

        for idx in trange(0, self.batch_idxs):
            self.t += 1
            # to feed real_data
            batch_files = list(zip(self.dataA[idx * self.args.batch_size:(idx + 1) * self.args.batch_size],
                                   self.dataB[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]))
            batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch_images).astype(np.float32)
            batch_images = ptu.tensor(batch_images)                                    # (b, 64, 84, 2)
            dataA_batch = batch_images[:,:,:,0].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
            dataB_batch = batch_images[:,:,:,1].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
            gaussian_noise = torch.abs(Normal(ptu.zeros((self.args.batch_size, 64, 84, 1)),
                                              ptu.ones((self.args.batch_size, 64, 84, 1)) * self.args.sigma_d).sample())

            # fake data
            dataA_batch_hat = self.generatorBA(dataB_batch)                            # (b, 64, 84, 1)
            dataB_batch_hat = self.generatorAB(dataA_batch)                            # (b, 64, 84, 1)
            dataA_batch_tail = self.generatorBA(dataB_batch_hat)                       # (b, 64, 84, 1)
            dataB_batch_tail = self.generatorAB(dataA_batch_hat)                       # (b, 64, 84, 1)
            dataA_batch_hat_sample = self.generatorBA(dataB_batch)
            dataB_batch_hat_sample = self.generatorAB(dataA_batch)

            # discriminator prediction
            DA_batch_real = self.discriminatorA(dataA_batch + gaussian_noise)          # (b, 16, 21, 1)
            DB_batch_real = self.discriminatorB(dataB_batch + gaussian_noise)          # (b, 16, 21, 1)
            DA_batch_fake = self.discriminatorA(dataA_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
            DB_batch_fake = self.discriminatorB(dataB_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
            DA_batch_fake_sample = self.discriminatorA(dataA_batch_hat_sample + gaussian_noise)
            DB_batch_fake_sample = self.discriminatorB(dataB_batch_hat_sample + gaussian_noise)

            # generator loss
            cycle_loss = self.args.L1_lambda * self.abs_criterion(dataA_batch, dataA_batch_tail) + \
                            self.args.L1_lambda * self.abs_criterion(dataB_batch, dataB_batch_tail)
            g_loss_a2b = self.criterionGAN(DB_batch_fake.detach(), ptu.ones(DB_batch_fake.shape))
            g_loss_b2a = self.criterionGAN(DA_batch_fake.detach(), ptu.ones(DA_batch_fake.shape))
            g_loss = (g_loss_a2b + g_loss_b2a).mean() + cycle_loss.mean()
            self.generatorAB_optimizer.zero_grad()
            self.generatorBA_optimizer.zero_grad()

            # discriminator loss
            da_loss_real = self.criterionGAN(DA_batch_real, ptu.ones(DA_batch_real.shape))
            da_loss_fake = self.criterionGAN(DA_batch_fake_sample, ptu.zeros(DA_batch_fake.shape))
            da_loss = (da_loss_real + da_loss_fake) / 2
            db_loss_real = self.criterionGAN(DB_batch_real, ptu.ones(DB_batch_real.shape))
            db_loss_fake = self.criterionGAN(DB_batch_fake_sample, ptu.zeros(DB_batch_fake.shape))
            db_loss = (db_loss_real + db_loss_fake) / 2
            d_loss = (da_loss + db_loss).mean()
            self.discriminatorA_optimizer.zero_grad()
            self.discriminatorB_optimizer.zero_grad()

            # backward
            g_loss.backward()
            d_loss.backward()
            self.generatorAB_optimizer.step()
            self.generatorBA_optimizer.step()
            self.discriminatorA_optimizer.step()
            self.discriminatorB_optimizer.step()

            with torch.no_grad():
                self.train_stats_writer.write("g_loss", self.t, g_loss.item())
                self.train_stats_writer.write("d_loss", self.t, d_loss.item())


    def test(self):
        with torch.no_grad():
            # shuffle training data
            np.random.shuffle(self.dataA_test)
            np.random.shuffle(self.dataB_test)
            print("Evaluation")
            g_loss_total = 0
            d_loss_total = 0
            for idx in trange(0, self.batch_idxs_test):
                # to feed real_data
                batch_files = list(zip(self.dataA_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size],
                                       self.dataB_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]))
                batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                batch_images = ptu.tensor(batch_images)                                    # (b, 64, 84, 2)
                dataA_batch = batch_images[:,:,:,0].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                dataB_batch = batch_images[:,:,:,1].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                gaussian_noise = torch.abs(Normal(ptu.zeros((self.args.batch_size, 64, 84, 1)),
                                                  ptu.ones((self.args.batch_size, 64, 84, 1)) * self.args.sigma_d).sample())

                # fake data
                dataA_batch_hat = self.generatorBA(dataB_batch)                            # (b, 64, 84, 1)
                dataB_batch_hat = self.generatorAB(dataA_batch)                            # (b, 64, 84, 1)
                dataA_batch_tail = self.generatorBA(dataB_batch_hat)                       # (b, 64, 84, 1)
                dataB_batch_tail = self.generatorAB(dataA_batch_hat)                       # (b, 64, 84, 1)
                dataA_batch_hat_sample = self.generatorBA(dataB_batch)
                dataB_batch_hat_sample = self.generatorAB(dataA_batch)

                # discriminator prediction
                DA_batch_real = self.discriminatorA(dataA_batch + gaussian_noise)          # (b, 16, 21, 1)
                DB_batch_real = self.discriminatorB(dataB_batch + gaussian_noise)          # (b, 16, 21, 1)
                DA_batch_fake = self.discriminatorA(dataA_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
                DB_batch_fake = self.discriminatorB(dataB_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
                DA_batch_fake_sample = self.discriminatorA(dataA_batch_hat_sample + gaussian_noise)
                DB_batch_fake_sample = self.discriminatorB(dataB_batch_hat_sample + gaussian_noise)

                # generator loss
                cycle_loss = self.args.L1_lambda * self.abs_criterion(dataA_batch, dataA_batch_tail) + \
                                self.args.L1_lambda * self.abs_criterion(dataB_batch, dataB_batch_tail)
                g_loss_a2b = self.criterionGAN(DB_batch_fake.detach(), ptu.ones(DB_batch_fake.shape))
                g_loss_b2a = self.criterionGAN(DA_batch_fake.detach(), ptu.ones(DA_batch_fake.shape))
                g_loss = (g_loss_a2b + g_loss_b2a).mean() + cycle_loss.mean()
                g_loss_total += g_loss.item()

                # discriminator loss
                da_loss_real = self.criterionGAN(DA_batch_real, ptu.ones(DA_batch_real.shape))
                da_loss_fake = self.criterionGAN(DA_batch_fake_sample, ptu.zeros(DA_batch_fake.shape))
                da_loss = (da_loss_real + da_loss_fake) / 2
                db_loss_real = self.criterionGAN(DB_batch_real, ptu.ones(DB_batch_real.shape))
                db_loss_fake = self.criterionGAN(DB_batch_fake_sample, ptu.zeros(DB_batch_fake.shape))
                db_loss = (db_loss_real + db_loss_fake) / 2
                d_loss = (da_loss + db_loss).mean()
                d_loss_total += d_loss.item()

            self.test_stats_writer.write("g_loss", self.t, g_loss_total / self.batch_idxs_test)
            self.test_stats_writer.write("d_loss", self.t, d_loss_total / self.batch_idxs_test)
            print("g_loss:", g_loss_total / self.batch_idxs_test)
            print("d_loss:", d_loss_total / self.batch_idxs_test)


    def save(self, name='trainer'):
        """ save this trainer """
        torch.save(self, os.path.join(self.args.directory, f'{name}_{self.t}.pth'))

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return torch.load(f)
