import logging
import numpy as np
import torch
from tqdm import tqdm
from utils import KL_loss, attention_diff


class FewShotKT():

    def __init__(self, dataloader, student_model, teacher_model, log_num):

        self.student_optimizer = torch.optim.SGD(momentum=0.9, nesterov=True)
        self.datalaoder = dataloader
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Todo Implment pytorch scheduler to reduce learning rate

        self.student_model = student_model
        self.teacher_model = teacher_model

        self.log_num = log_num
        self.num_epochs = self.calculate_epochs()

        # Defining train_kd & train_and_evaluate_kd functions

    def train_KT_AT(self):
        """

        :param student_model:
        :param teacher_model:
        :param num_epochs:
        :param dataloader:
        :return:
        """

        # summary for current training loop and a running average object for loss
        # Use tqdm for progress bar

        for epoch in tqdm(range(self.num_epochs)):
            self.train()

            if epoch % self.log_num:
                print("logging")

    def train(self):

        for i, (train_batch, labels_batch) in enumerate(self.dataloader):
            self.student_optimizer.zero_grad()

            # move to GPU if available
            train_batch, labels_batch = train_batch.to(self.device), labels_batch.to(self.device)

            # compute model output, fetch teacher output, and compute KD loss
            student_logits, student_activations = self.student_model(train_batch)

            # get one batch output from teacher_outputs list
            teacher_logits, teacher_activations = self.teacher_model(train_batch)


            KL_loss = KL_loss(teacher_logits, student_logits)
            attention_loss = attention_diff()
            loss = KL_loss + attention_loss

            acc = self.accuracy()

            loss.backward()

            # performs updates using calculated gradients
            self.student_optimizer.step()

    def test(self):
        pass



    def accuracy(self):
        return 0

    def calculate_epochs(self):
        return 0

if __name__ == '__main__':
    # fix random seed for reproducibility
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
