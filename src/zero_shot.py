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
import config
import dataloaders
import EfficientNet


class ZeroShot:
    def __init__(self):
        self.ng = 1
        self.ns = 10
        self.counter = 0
        self.acc_counter = 0
        self.log_num = 1000
        self.num_epochs = 80000

        self.dataset = config.dataset

        _, self.testloader, _, self.num_classes = dataloaders.transform_data(self.dataset, M=config.downsample['value'],
                                                                             down=config.downsample['action'])

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_type = config.model_type

        if self.model_type == "rnn":
            self.teacher_model = utils.load_teacher_rnn()
            teacher_path = f"{config.save_path}/{self.dataset}-no_teacher-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-seed{config.seed}.pth"

            if os.path.exists(teacher_path):
                checkpoint = torch.load(teacher_path, map_location=self.device)
            else:
                raise ValueError('No file with the pretrained model selected')

            self.teacher_model.load_state_dict(checkpoint)
            
            self.student_model = utils.load_student_rnn()
            if config.downsample['action']:
                self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-down_sample{config.downsample['value']}-seed{config.seed}.pth"
            else:
                self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-seed{config.seed}.pth"
        
        elif self.model_type == "efficient_net":
            self.teacher_model = EfficientNet.EfficientNet(config.teacher_efficient_net['input_features'], config.teacher_efficient_net['model'])

            teacher_path = f"{config.save_path}/{self.dataset}-no_teacher-efficient_net-seed{config.seed}.pth"

            if os.path.exists(teacher_path):
                checkpoint = torch.load(teacher_path, map_location=self.device)
            else:
                raise ValueError('No file with the pretrained model selected')

            self.teacher_model.load_state_dict(checkpoint)
            
            self.student_model = EfficientNet.EfficientNet(config.student_efficient_net['input_features'], config.student_efficient_net['model'])

            if config.downsample['action']:
                self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-efficient_net_student-down_sample{config.downsample['value']}-seed{config.seed}.pth"
            else:
                self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-efficient_net_student-seed{config.seed}.pth"
        else:
            raise ValueError('Invalid model type')

        
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        self.student_model.to(self.device)
        self.student_model.train()

        # Load teacher and initialise student network
        self.student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=2e-3)
        self.cosine_annealing_student = optim.lr_scheduler.CosineAnnealingLR(self.student_optimizer, self.num_epochs)

        self.generator = Generator.Generator(100)
        self.generator.to(self.device)
        self.generator.train()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
        self.cosine_annealing_generator = optim.lr_scheduler.CosineAnnealingLR(self.generator_optimizer,
                                                                               self.num_epochs)

        
        self.generator_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-generator-seed{config.seed}.pth"

    def train(self):
        best_acc = 0
        accuracy_dict = {}

        for batch in tqdm(range(self.num_epochs)):

            for _ in range(self.ng):
                self.generator_optimizer.zero_grad()
                # generate guassian noise
                z = torch.randn((128, 100)).to(self.device)

                # get generator output
                psuedo_datapoint = self.generator(z)

                # teacher/student outputs: logits, attention1, attention2, attention3
                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits = self.student_model(psuedo_datapoint)[0]
                teacher_logits = self.teacher_model(psuedo_datapoint)[0]

                generator_loss = -(utils.KL_Loss(student_logits, teacher_logits))
                generator_loss.backward()

                # performs updates using calculated gradients
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
                self.generator_optimizer.step()

            psuedo_datapoint = psuedo_datapoint.detach()

            with torch.no_grad():
                teacher_outputs = self.teacher_model(psuedo_datapoint)

            for _ in range(self.ns):
                self.student_optimizer.zero_grad()

                # teacher/student outputs: logits, attention1, attention2, attention3
                # compute model output, fetch teacher/student output, and compute KD loss
                student_logits = self.student_model(psuedo_datapoint)

                # student_loss = KL_Loss(teacher_logits, teacher_outputs)
                student_loss = utils.student_loss_zero_shot(student_logits, teacher_outputs)

                student_loss.backward()
                # performs updates using calculated gradients
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 5)
                self.student_optimizer.step()

            if (batch + 1) % self.log_num == 0 or (batch + 1) == self.num_epochs:
                acc = self.test()

                print(f"\nAccuracy: {acc:05.3f}")
                print(f'Student Loss: {student_loss:05.3f}')
                utils.writeMetrics({"accuracy": acc}, self.acc_counter)
                accuracy_dict[batch] = acc
                utils.log_accuracy("zero_shot.csv", accuracy_dict)
                self.acc_counter += 1
                self.save_model()

                if acc > best_acc:
                    best_acc = acc

            utils.writeMetrics({"Student Loss": student_loss, "Generator Loss": generator_loss}, self.counter)
            self.counter += 1
            self.cosine_annealing_generator.step()
            self.cosine_annealing_student.step()

    def test(self, test=False):
        if test == True:
            if os.path.exists(self.student_save_path):
                checkpoint = torch.load(self.student_save_path, map_location=self.device)
            else:
                raise ValueError('No file with the pretrained model selected')

            self.student_model.load_state_dict(checkpoint)
        self.student_model.eval()

        running_acc =  0

        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                student_logits, *student_activations = self.student_model(data)

                running_acc += utils.accuracy(student_logits.data, label)

        final_accuracy = running_acc / len(self.testloader)

        if test == True:
            print(f"Test accuracy = {final_accuracy}")
        
        return final_accuracy

    def save_model(self):
        torch.save(self.student_model.state_dict(), self.student_save_path)
        torch.save(self.generator.state_dict(), self.generator_save_path)

