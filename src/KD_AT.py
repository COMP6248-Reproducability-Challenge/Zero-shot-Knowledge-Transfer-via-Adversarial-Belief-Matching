import logging
import numpy as np
import torch
from tqdm import tqdm
from utils import KL_AT_loss, accuracy, log_accuracy, plot_accuracy, writeMetrics
import os
import ResNet
from torch import optim
import dataloaders
import config

class FewShotKT:
    def __init__(self):

        self.dataset = config.dataset
        self.M = config.downsample['value']
        self.trainloader, self.testloader, self.validationloader, self.num_classes = dataloaders.transform_data(self.dataset, 
                                                    M= config.downsample['value'], down= config.downsample['action'])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.teacher_model = ResNet.WideResNet(depth=config.teacher['depth'], num_classes=self.num_classes, widen_factor=config.teacher['widen_factor'], 
                    input_features=config.teacher['input_features'], output_features=config.teacher['output_features'], 
                    dropRate=config.teacher['dropRate'], strides=config.teacher['strides'])
        self.teacher_model.to(self.device)

        teacher_path = f"{config.save_path}/{self.dataset}-no_teacher-wrn-{config.teacher['depth']}-{config.teacher['widen_factor']}-{config.teacher['dropRate']}-seed{config.seed}.pth"
        
        if os.path.exists(teacher_path):
            checkpoint = torch.load(teacher_path, map_location=self.device)
        else:
            raise ValueError('No file with the pretrained model selected')

        self.teacher_model.load_state_dict(checkpoint)
        self.teacher_model.eval()

        self.student_model = ResNet.WideResNet(depth=config.student['depth'], num_classes=self.num_classes, widen_factor=config.student['widen_factor'], 
                    input_features=config.student['input_features'], output_features=config.student['output_features'], 
                    dropRate=config.student['dropRate'], strides=config.student['strides'])
        self.student_model.to(self.device)
        self.student_model.train()

        self.log_num = 10
        self.num_epochs = self.calculate_epochs()
        self.counter = 0

        self.student_optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.student_optimizer, milestones=[0.3*self.num_epochs - 1,0.6*self.num_epochs - 1,0.8*self.num_epochs - 1], gamma=0.2)

        if config.downsample['action']:
            self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher['depth']}-{config.teacher['widen_factor']}-{config.teacher['dropRate']}-down_sample{config.downsample['value']}-seed{config.seed}.pth"
        else:
            self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher['depth']}-{config.teacher['widen_factor']}-{config.teacher['dropRate']}-seed{config.seed}.pth"

    def train_KT_AT(self):
        # summary for current training loop and a running average object for loss
        # Use tqdm for progress bar
        accuracy_dict = {}

        
        for epoch in range(self.num_epochs):
            self.student_model.train()
            self.train()

            if epoch % self.log_num == 0:
                acc = self.test(epoch)
                accuracy_dict[epoch] = acc
                torch.save(self.student_model.state_dict(), self.save_path)
            
            self.scheduler.step()

        log_accuracy("KD_AT.csv", accuracy_dict)
        plot_accuracy("KD_AT.csv")

    def train(self):
        running_acc = running_loss = 0

        with tqdm(self.trainloader, total=len(self.trainloader), desc='train', position=0, leave=True) as t:
            for batch_num, input in enumerate(self.trainloader):
                self.student_optimizer.zero_grad()

                # move to GPU if available
                train_batch, labels_batch = input
                train_batch, labels_batch = train_batch.to(self.device), labels_batch.to(self.device)

                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits, *student_activations = self.student_model(train_batch)
                teacher_logits, *teacher_activations = self.teacher_model(train_batch)

                # teacher/student outputs: logits, attention1, attention2, attention3

                loss = KL_AT_loss(teacher_logits, student_logits, student_activations, teacher_activations, labels_batch)
                loss.backward()

                running_loss += loss.data
                running_acc += accuracy(student_logits.data, labels_batch)
                writeMetrics({"accuracy": running_acc/(batch_num+1),
                              "loss": running_loss/(batch_num+1)}, self.counter)
                self.counter +=1

                t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(batch_num+1)), loss='{:05.3f}'.format(running_loss/(batch_num+1)))
                t.update()
                # performs updates using calculated gradients
                self.student_optimizer.step()

    def test(self, epoch):
        self.student_model.eval()

        running_acc = 0
        count = len(self.testloader)
        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                student_logits, *student_activations = self.student_model(data)
                teacher_logits, *teacher_activations = self.teacher_model(data)

                running_acc += accuracy(student_logits.data, label)

        print(f"Test accuracy: {running_acc/len(self.testloader)}")
        return (running_acc/count)

    def calculate_epochs(self):
        if self.dataset == 'cifar10':
            num_epochs = int(200 * 50000 / (10 * self.M))
        else:
            epochs = int(73257 * 100/ (10 * self.M))
        return num_epochs

    def save_model(self):
        pass
