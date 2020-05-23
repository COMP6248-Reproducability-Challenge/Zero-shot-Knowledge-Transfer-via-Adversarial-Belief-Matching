import logging
import numpy as np
import torch
from tqdm import tqdm
import utils
import ResNet
from torch import optim
from dataloaders import transform_data
import Generator

class ZeroShot:
    def __init__(self, M, dataset_name, save_path, seed):
        self.ng = 1
        self.ns = 10
        self.M = M
        self.total_batches = 65000
        self.seed = seed
        self.dataset_name = dataset_name
        self.save_path = save_path

        _, self.testloader, _, self.num_classes = transform_data(self.dataset_name, self.M)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        strides = [1, 2, 2]
        self.teacher_model = ResNet.WideResNet(depth=40, num_classes=self.num_classes, widen_factor=2, input_features=3,
                    output_features=16, dropRate=0.0, strides=strides)
        torch_checkpoint = torch.load(f'../PreTrainedModels/{self.dataset_name}-no_teacher-wrn-40-2-0.0-seed{self.seed}.pth', 
                        map_location=self.device)
        self.teacher_model.load_state_dict(torch_checkpoint)
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()

        self.student_depth = 16
        self.dropRate = 0.0
        self.widen_factor = 1

        self.student_model = ResNet.WideResNet(depth=self.student_depth, num_classes=self.num_classes, widen_factor=self.widen_factor, 
                                        input_features=3,output_features=16, dropRate=self.dropRate, strides=strides)
        self.student_model = self.student_model.to(self.device)
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
