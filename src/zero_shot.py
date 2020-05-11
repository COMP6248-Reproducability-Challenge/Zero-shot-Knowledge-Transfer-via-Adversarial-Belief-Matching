import logging
import numpy as np
import torch
from tqdm import tqdm
from utils import KL_AT_loss, accuracy, KL_Loss
from WRN_temp import WideResNet
from torch import optim
from dataloaders import transform_data
import Generator

class ZeroShot:
    def __init__(self, M, dataset_name):
        self.ng = 1
        self.ns = 10
        self.total_batches = 65000
        self.dataset_name = dataset_name
        _, self.testloader, _, _ = transform_data(self.dataset_name, self.M)

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
        self.student_optimizer = torch.optim.Adam(student_model.parameters(), lr=2e-3)
        self.cosine_annealing_student = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, total_batches)

        self.generator = Generator()
        self.generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
        self.cosine_annealing_generator = optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, total_batches)

        self.log_num = 10
        self.numGeneratorIterations = 5000


    def train_ZS(self):

        for batch in tqdm(range(self.total_batches)):

            # generate guassian noise
            z = torch.randn((128, 100)).to(device)

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
                z = torch.randn((128, 100)).to(device)

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

            # calclate accuracy on test set here
            acc = accuracy(
                    model =  self.student_model,
                    data = self.test_loader,
                    device = self.device
                            )
            cosine_annealing_generator.step()
            cosine_annealing_student.step()


        print("finished")

    def test(self):
        """
            - call train, after some number if epochs, call accuracy
            - repeat
        """
        pass

    def save_model(self):
        pass
    
    
