import logging
import numpy as np
import torch
from tqdm import tqdm
from utils import KL_AT_loss, accuracy
from WRN_temp import WideResNet
from torch import optim
from dataloaders import transform_data

class FewShotKT:

    def __init__(self, M, dataset_name):


        self.M = M
        self.dataset_name = dataset_name
        self.trainloader, self.testloader, self.validationloader, _ = transform_data(self.dataset_name, self.M)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        strides = [1, 1, 2, 2]
        self.teacher_model = WideResNet(d=40, k=2, n_classes=10, input_features=3,
                                 output_features=16, strides=strides)
        self.teacher_model = self.teacher_model.to(self.device)
        torch_checkpoint = torch.load('wrn-40-2-seed-0-dict.pth', map_location=self.device)
        self.teacher_model.load_state_dict(torch_checkpoint)

        self.student_model = WideResNet(d=16, k=1, n_classes=10, input_features=3,
                                 output_features=16, strides=strides)
        self.student_model = self.student_model.to(self.device)
        self.student_model.train()
        # Load teacher and initialise student network

        self.student_optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.student_optimizer, milestones=[60, 120, 160], gamma=0.2)

        self.log_num = 10
        self.num_epochs = self.calculate_epochs()



    def train_KT_AT(self):
        """

        """

        # summary for current training loop and a running average object for loss
        # Use tqdm for progress bar

        self.teacher_model.eval()
        for epoch in tqdm(range(self.num_epochs)):
            self.train()

            if epoch % self.log_num == 0:
                self.test()

    def train(self):

        for i, input in enumerate(self.trainloader):
            self.student_optimizer.zero_grad()

            # move to GPU if available
            train_batch, labels_batch = input
            train_batch, labels_batch = train_batch.to(self.device), labels_batch.to(self.device)

            # compute model output, fetch teacher/student output, and compute KD loss
            student_logits, *student_activations = self.student_model(train_batch)
            teacher_logits, *teacher_activations = self.teacher_model(train_batch)

            # teacher/student outputs: logits, attention1, attention2, attention3

            loss = KL_AT_loss(teacher_logits, student_logits, student_activations, teacher_activations, labels_batch)


            acc = accuracy(self.student_model, self.testloader, self.device)
            print(f'Current accuracy is {acc}')

            loss.backward()

            # performs updates using calculated gradients
            self.student_optimizer.step()



        print("finished")

    def test(self):
        pass

    def calculate_epochs(self):
        num_epochs = 0
        if self.dataset_name == 'cifar10':
            num_epochs = int(200 * (5000 / self.M))
        return num_epochs

    def save_model(self):
        pass
