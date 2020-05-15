import logging
import numpy as np
import torch
from tqdm import tqdm
from utils import KL_AT_loss, accuracy, KL_Loss
import ResNet
from torch import optim
from dataloaders import transform_data
import Generator

class ZeroShot:
    def __init__(self, M, dataset_name):
        self.ng = 1
        self.ns = 10
        self.M = M
        self.total_batches = 65000
        self.dataset_name = dataset_name
        _, self.testloader, _, self.num_classes = transform_data(self.dataset_name, self.M)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        strides = [1, 2, 2]
        self.teacher_model = ResNet.WideResNet(depth=40, num_classes=self.num_classes, widen_factor=2, input_features=3,
                    output_features=16, dropRate=0.0, strides=strides)
        torch_checkpoint = torch.load('../PreTrainedModels/cifar10-no_teacher-wrn-40-2-0.0-seed0.pth', map_location=self.device)
        self.teacher_model.load_state_dict(torch_checkpoint['model_state_dict'])
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()

        self.student_model = ResNet.WideResNet(depth=16, num_classes=self.num_classes, widen_factor=1, input_features=3,
                                 output_features=16, dropRate=0.0, strides=strides)
        self.student_model = self.student_model.to(self.device)
        self.student_model.train()

        # Load teacher and initialise student network
        self.student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=2e-3)
        self.cosine_annealing_student = optim.lr_scheduler.CosineAnnealingLR(self.student_optimizer, self.total_batches)

        self.generator = Generator.Generator()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
        self.cosine_annealing_generator = optim.lr_scheduler.CosineAnnealingLR(self.generator_optimizer, self.total_batches)

        self.log_num = 10
        self.numGeneratorIterations = 5000
        self.num_epochs = 80000

    def train_ZS(self):
        for epoch in range(self.num_epochs):
            self.student_model.train()
            self.train()
            self.student_model.eval()
            self.test(epoch)

    def train(self):
        for batch in tqdm(range(self.total_batches)):
            # generate guassian noise
            z = torch.randn((128, 100)).to(self.device)

            for i in range(self.ng):
                self.generator_optimizer.zero_grad()

                # get generator output
                psuedo_datapoint = self.generator(z)

                # teacher/student outputs: logits, attention1, attention2, attention3
                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits = self.student_model(psuedo_datapoint)[0]
                teacher_logits = self.teacher_model(psuedo_datapoint)[0]

                generator_loss = - (KL_Loss(teacher_logits, student_logits))
                generator_loss.backward()

                # performs updates using calculated gradients
                self.generator_optimizer.step()

            for i in range(self.ns):
                self.student_optimizer.zero_grad()

                # generate guassian noise
                z = torch.randn((128, 100)).to(self.device)

                # get generator output
                psuedo_datapoint = self.generator(z)

                # teacher/student outputs: logits, attention1, attention2, attention3
                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits = self.student_model(psuedo_datapoint)[0]
                teacher_logits = self.teacher_model(psuedo_datapoint)[0]

                student_loss = KL_Loss(teacher_logits, student_logits)

                student_loss.backward()
                # performs updates using calculated gradients
                self.student_optimizer.step()

            self.cosine_annealing_generator.step()
            self.cosine_annealing_student.step()
    
    def test(self, epoch):
        running_acc = count = 0
        
        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                student_logits, *student_activations = self.student_model(data)
                teacher_logits, *teacher_activations = self.teacher_model(data)

                running_acc += accuracy(student_logits.data, label, self.device)
                count += 1
        
        print(f'Epoch {epoch + 1} accuracy: {running_acc/count}')
