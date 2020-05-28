import dataloaders
import ResNet
import torch
import torch.nn as nn
import utils
from tqdm import tqdm
import config
import EfficientNet
import os

class BeliefMatch:
    def __init__(self):
        self.dataset = config.dataset
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if config.mode == "kd_at":
            self.M = 200
            self.downsample = True
        elif config.mode == "zero_shot":
            self.M = 0
            self.downsample = False
        else:
            raise ValueError('Not valid mode')

        _, self.testloader, _, self.num_classes = dataloaders.transform_data(self.dataset, test_batch_size= 1, 
                                                            M=self.M, down=self.downsample)

        self.teacher_model = ResNet.WideResNet(depth=config.teacher_rnn['depth'], num_classes=self.num_classes,
                                               widen_factor=config.teacher_rnn['widen_factor'],
                                               input_features=config.teacher_rnn['input_features'],
                                               output_features=config.teacher_rnn['output_features'],
                                               dropRate=config.teacher_rnn['dropRate'],
                                               strides=config.teacher_rnn['strides'])

        teacher_path = f"{config.save_path}/{self.dataset}-no_teacher-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-seed{config.seed}.pth"

        if os.path.exists(teacher_path):
            checkpoint = torch.load(teacher_path, map_location=self.device)
        else:
            raise ValueError('No file with the pretrained model selected')

        self.teacher_model.load_state_dict(checkpoint)
        
        self.student_model = ResNet.WideResNet(depth=config.student_rnn['depth'], num_classes=self.num_classes,
                                            widen_factor=config.student_rnn['widen_factor'],
                                            input_features=config.student_rnn['input_features'],
                                            output_features=config.student_rnn['output_features'],
                                            dropRate=config.student_rnn['dropRate'],
                                            strides=config.student_rnn['strides'])

        if self.downsample:
            student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-down_sample{self.M}-seed{config.seed}.pth"
        else:
            student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-seed{config.seed}.pth"

        if os.path.exists(student_save_path):
            checkpoint = torch.load(student_save_path, map_location=self.device)
        else:
            raise ValueError('No file with the pretrained model selected')

        self.student_model.load_state_dict(checkpoint)
        self.teacher_model.eval()
        self.student_model.eval()

        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
    
    def calculate(self):
        xi = 1
        K = 100
        N = 1000
        C = 10
        criterion = nn.CrossEntropyLoss()

        count = 0
        mte = 0 #mean transition error
        with tqdm(total=N, position=0, leave=True) as t: 
            for image, label in self.testloader:
                if count == N:
                    break
                
                image, label = image.to(self.device), label.to(self.device)

                teacher_output, *_ = self.teacher_model(image)
                _, teacher_prediction = torch.max(teacher_output.data, 1)

                student_output, *_ = self.student_model(image)
                _, student_prediction = torch.max(student_output.data, 1)

                if teacher_prediction != student_prediction or student_prediction != label:
                    continue

                count += 1
                mte_n = 0

                for other_label in range(0,10):
                    if other_label == label:
                        continue
                    
                    other_label = torch.Tensor([other_label]).long().to(self.device)

                    image_adv = image.detach().clone()
                    image_adv.requires_grad = True
                    image_adv = image_adv.to(self.device)

                    mte_k = 0
                    for _ in range(K):
                        self.teacher_model.zero_grad()

                        teacher_output, *_ = self.teacher_model(image_adv)
                        with torch.no_grad():
                            student_output, *_ = self.student_model(image_adv)

                        loss = criterion(teacher_output, other_label)
                        loss.backward()

                        image_adv.data -= xi * image_adv.grad.data
                        image_adv.grad.data.zero_()

                        pj_teacher = nn.functional.softmax(teacher_output, 1)[0][other_label].item()
                        pj_student = nn.functional.softmax(student_output, 1)[0][other_label].item()

                        mte_k += abs(pj_teacher - pj_student)
                
                    mte_n += mte_k/K
                
                mte += mte_n/(C-1)
                t.update()

        mte /= N
        print(f"Mean Transition Error for {config.mode} on {self.dataset} = {mte}")







