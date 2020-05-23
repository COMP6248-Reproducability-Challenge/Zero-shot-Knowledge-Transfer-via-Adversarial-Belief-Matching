import logging
import numpy as np
import torch
from tqdm import tqdm
from utils import KL_AT_loss, accuracy, log_accuracy, plot_accuracy
import ResNet
from torch import optim
from dataloaders import transform_data

class FewShotKT:
    def __init__(self, M, dataset_name):
        self.M = M
        self.dataset_name = dataset_name
        self.trainloader, self.testloader, self.validationloader, self.num_classes = transform_data(self.dataset_name, self.M)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        strides = [1, 2, 2]

        self.teacher_model = ResNet.WideResNet(depth=40, num_classes=self.num_classes, widen_factor=2, input_features=3,
                    output_features=16, dropRate=0.0, strides=strides)
        self.teacher_model = self.teacher_model.to(self.device)
        torch_checkpoint = torch.load('../PreTrainedModels/cifar10-no_teacher-wrn-40-2-0.0-seed0.pth', map_location=self.device)
        self.teacher_model.load_state_dict(torch_checkpoint)
        self.teacher_model.eval()

        #Check teacher accuracy
        self.verify_teacher_accuracy()

        self.student_model = ResNet.WideResNet(depth=16, num_classes=self.num_classes, widen_factor=1, input_features=3,
                                 output_features=16,dropRate=0.0, strides=strides)
        self.student_model = self.student_model.to(self.device)
        self.student_model.train()
        # Load teacher and initialise student network

        self.student_optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.student_optimizer, milestones=[60, 120, 160], gamma=0.2)

        self.log_num = 10
        self.num_epochs = self.calculate_epochs()



    def train_KT_AT(self):
        """

        """

        # summary for current training loop and a running average object for loss
        # Use tqdm for progress bar
        accuracy_dict = {}

        for epoch in range(self.num_epochs):
            self.student_model.train()
            if epoch in [60,120,160]:
                for param_group in self.student_optimizer.param_groups:
                    self.student_optimizer.param_group["lr"] /= 5
            self.train(epoch)

            if epoch % self.log_num == 0:
                acc = self.test(epoch)
                accuracy_dict[epoch] = acc

        log_accuracy("KD_AT.csv", accuracy_dict)
        plot_accuracy("KD_AT.csv")

    def train(self, epoch):
        running_acc = count = 0

        with tqdm(self.trainloader, total=len(self.trainloader), desc='train', position=0, leave=True) as t:
            for input in self.trainloader:
                self.student_optimizer.zero_grad()

                # move to GPU if available
                train_batch, labels_batch = input
                train_batch, labels_batch = train_batch.to(self.device), labels_batch.to(self.device)

                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits, *student_activations = self.student_model(train_batch)
                teacher_logits, *teacher_activations = self.teacher_model(train_batch)

                # teacher/student outputs: logits, attention1, attention2, attention3

                loss = KL_AT_loss(teacher_logits, student_logits, student_activations, teacher_activations, labels_batch)

                running_acc += accuracy(student_logits.data, labels_batch)
                count += 1

                loss.backward()
                t.set_postfix(accuracy='{:05.3f}'.format(running_acc/count))
                t.update()
                # performs updates using calculated gradients
                self.student_optimizer.step()

        print(f'Epoch {epoch + 1} accuracy: {running_acc/count}')

    def test(self, epoch):
        print("     Started Testing     ")
        print("##########################")
        self.student_model.eval()

        running_acc = 0

        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                student_logits, *student_activations = self.student_model(data)
                teacher_logits, *teacher_activations = self.teacher_model(data)

                running_acc += accuracy(student_logits.data, label)

        print(f"Test accuracy: {running_acc/len(self.testloader)}")
        print("##########################")
        print("     Ended Testing     ")
        return (running_acc/count)

    def calculate_epochs(self):
        num_epochs = 0
        if self.dataset_name == 'cifar10':
            num_epochs = int(200 * (5000 / self.M))
        else:
            epochs = int(5000 * 100/ self.M)
        return num_epochs

    def verify_teacher_accuracy(self):
        running_acc = 0

        #with torch.no_grad():
        #     for data, label in self.trainloader:
        #         data, label = data.to(self.device), label.to(self.device)

        #         teacher_logits, *teacher_activations = self.teacher_model(data)

        #         running_acc += accuracy(teacher_logits.data, label)

        # print(f"Teacher accuracy on Training data: {running_acc/len(self.trainloader)}")

        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                teacher_logits, *teacher_activations = self.teacher_model(data)

                running_acc += accuracy(teacher_logits.data, label)

        print(f"Teacher accuracy on Testing data: {running_acc/len(self.testloader)}")

    def save_model(self):
        pass
