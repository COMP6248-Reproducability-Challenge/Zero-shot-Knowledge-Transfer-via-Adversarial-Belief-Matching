import logging
import numpy as np
import torch
from tqdm import tqdm
from utils import *
import ResNet
from torch import optim
from dataloaders import transform_data
import Generator
import os
import config
import dataloaders


class ZeroShot:
    def __init__(self):
        self.ng = 1
        self.ns = 10
        self.counter = 0
        self.acc_counter = 0
        self.log_num = 10
        self.num_epochs = 80000


        self.dataset = config.dataset

        _, self.testloader, _, self.num_classes = dataloaders.transform_data(self.dataset, M=config.downsample['value'],
                                                                             down=config.downsample['action'])

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.teacher_model = ResNet.WideResNet(depth=config.teacher_rnn['depth'], num_classes=self.num_classes,
                                               widen_factor=config.teacher_rnn['widen_factor'],
                                               input_features=config.teacher_rnn['input_features'],
                                               output_features=config.teacher_rnn['output_features'],
                                               dropRate=config.teacher_rnn['dropRate'], strides=config.teacher_rnn['strides'])
        self.teacher_model.to(self.device)
        
        teacher_path = f"{config.save_path}/{self.dataset}-no_teacher-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-seed{config.seed}.pth"

        if os.path.exists(teacher_path):
            checkpoint = torch.load(teacher_path, map_location=self.device)
        else:
            raise ValueError('No file with the pretrained model selected')

        self.teacher_model.load_state_dict(checkpoint)
        self.teacher_model.eval()

        self.student_model = ResNet.WideResNet(depth=config.student_rnn['depth'], num_classes=self.num_classes,
                                               widen_factor=config.student_rnn['widen_factor'],
                                               input_features=config.student_rnn['input_features'],
                                               output_features=config.student_rnn['output_features'],
                                               dropRate=config.student_rnn['dropRate'], strides=config.student['strides'])
        self.student_model.to(self.device)
        self.student_model.train()

        # Load teacher and initialise student network
        self.student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=2e-3)
        self.cosine_annealing_student = optim.lr_scheduler.CosineAnnealingLR(self.student_optimizer, self.total_batches)

        self.generator = Generator.Generator(100)
        self.generator.to(self.device)
        self.generator.train()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
        self.cosine_annealing_generator = optim.lr_scheduler.CosineAnnealingLR(self.generator_optimizer,
                                                                               self.total_batches)



        if config.downsample['action']:
            self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-down_sample{config.downsample['value']}-seed{config.seed}.pth"
            self.generator_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-generator-down_sample{config.downsample['value']}-seed{config.seed}.pth"
        else:
            self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-seed{config.seed}.pth"
            self.generator_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-generator-seed{config.seed}.pth"

    def train(self):
        best_acc = 0

        for batch in tqdm(range(self.num_epochs)):

            for i in range(self.ng):
                self.generator_optimizer.zero_grad()
                # generate guassian noise
                z = torch.randn((128, 100)).to(self.device)

                # get generator output
                psuedo_datapoint = self.generator(z)

                # teacher/student outputs: logits, attention1, attention2, attention3
                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits = self.student_model(psuedo_datapoint)[0]
                teacher_logits = self.teacher_model(psuedo_datapoint)[0]

                generator_loss = -(KL_Loss(teacher_logits, student_logits))
                generator_loss.backward()

                # performs updates using calculated gradients
                self.generator_optimizer.step()

            psuedo_datapoint = psuedo_datapoint.detach()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(psuedo_datapoint)

            for i in range(self.ns):
                self.student_optimizer.zero_grad()

                # generate guassian noise
                # z = torch.randn((128, 100)).to(self.device)

                # get generator output
                # psuedo_datapoint = self.generator(z)

                # teacher/student outputs: logits, attention1, attention2, attention3
                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits = self.student_model(psuedo_datapoint)

                # student_loss = KL_Loss(teacher_logits, teacher_outputs)
                student_loss = student_loss_zero_shot(student_logits, teacher_outputs)

                student_loss.backward()
                # performs updates using calculated gradients
                self.student_optimizer.step()

            if (batch + 1) % 10 == 0:
                acc = self.test()

                print(f"\nAccuracy: {acc:05.3f}")
                print(f'Student Loss: {student_loss:05.3f}')
                writeMetrics({"accuracy": acc}, self.acc_counter)
                self.acc_counter += 1

                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.student_model.state_dict(), self.student_save_path)
                    torch.save(self.generator.state_dict(), self.generator_save_path)
                    best_acc = acc

            writeMetrics({"Student Loss": student_loss, "Generator Loss": generator_loss}, self.counter)
            self.counter += 1
            self.cosine_annealing_generator.step()
            self.cosine_annealing_student.step()

    def test(self):
        running_acc = count = 0

        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                student_logits, *student_activations = self.student_model(data)
                teacher_logits, *teacher_activations = self.teacher_model(data)

                running_acc += accuracy(student_logits.data, label)
                count += 1

        return running_acc / len(self.testloader)
