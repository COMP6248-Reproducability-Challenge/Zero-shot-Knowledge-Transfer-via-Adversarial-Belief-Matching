import logging
import numpy as np
import torch
from tqdm import tqdm
import utils
import ResNet
from torch import optim
from dataloaders import transform_data
import Generator
import os

class ZeroShot:
    def __init__(self):
        self.ng = 1
        self.ns = 10

        self.total_batches = 65000
        self.dataset_name = dataset_name

        _, self.testloader, _ , self.num_classes = dataloaders.transform_data(self.dataset, M= config.downsample['value'], down= config.downsample['action'])

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

        # Load teacher and initialise student network
        self.student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=2e-3)
        self.cosine_annealing_student = optim.lr_scheduler.CosineAnnealingLR(self.student_optimizer, self.total_batches)

        self.generator = Generator.Generator(100)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
        self.cosine_annealing_generator = optim.lr_scheduler.CosineAnnealingLR(self.generator_optimizer, self.total_batches)

        self.log_num = 10
        self.numGeneratorIterations = 5000
        self.num_epochs = 80000

        self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.student['depth']}-{config.student['widen_factor']}-{config.student['dropRate']}-seed{config.seed}.pth"
        self.generator_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-generator-seed{config.seed}.pth"

    def train(self):
        best_acc = 0

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

                generator_loss = -(utils.KL_Loss(teacher_logits, student_logits))
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

                student_loss = utils.KL_Loss(teacher_logits, student_logits)

                student_loss.backward()
                # performs updates using calculated gradients
                self.student_optimizer.step()

            if (batch + 1) % 500 == 0:
                acc = self.test()

                if acc > best_acc:
                    best_acc = acc

                    torch.save(self.student_model.state_dict(), self.student_save_path)
                    torch.save(self.generator.state_dict(), self.generator_save_path)
                    best_acc = acc

            self.cosine_annealing_generator.step()
            self.cosine_annealing_student.step()
    
    def test(self):
        running_acc = count = 0
        
        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                student_logits, *student_activations = self.student_model(data)
                teacher_logits, *teacher_activations = self.teacher_model(data)

                running_acc += utils.accuracy(student_logits.data, label)
                count += 1
        
        return running_acc/len(self.test_loader)
