import os

import torch
from torch import optim
from tqdm import tqdm
import utils
import os
from torch import optim
import dataloaders
import config
import dataloaders
import utils
import EfficientNet


class FewShotKT:
    def __init__(self):
        self.dataset = config.dataset
        self.M = config.downsample['value']
        self.trainloader, self.testloader, self.validationloader, self.num_classes = dataloaders.transform_data(
            self.dataset,
            M=config.downsample['value'], down=config.downsample['action'])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_type = config.model_type

        if self.model_type == "rnn":
            self.teacher_model = utils.load_teacher_rnn(self.num_classes)
            teacher_path = f"{config.save_path}/{self.dataset}-no_teacher-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-seed{config.seed}.pth"

            if os.path.exists(teacher_path):
                checkpoint = torch.load(teacher_path, map_location=self.device)
            else:
                raise ValueError('No file with the pretrained model selected')

            self.teacher_model.load_state_dict(checkpoint)
            
            self.student_model = utils.load_student_rnn(self.num_classes)
            if config.downsample['action']:
                self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-down_sample{config.downsample['value']}-seed{config.seed}.pth"
            else:
                self.student_save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn_student-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.student_rnn['depth']}-{config.student_rnn['widen_factor']}-{config.student_rnn['dropRate']}-seed{config.seed}.pth"

        elif self.model_type == "efficient_net":
            self.teacher_model = EfficientNet.EfficientNet(config.teacher_efficient_net['input_features'],
                                                           config.teacher_efficient_net['model'])

            teacher_path = f"{config.save_path}/{self.dataset}-no_teacher-efficient_net-seed{config.seed}.pth"

            if os.path.exists(teacher_path):
                checkpoint = torch.load(teacher_path, map_location=self.device)
            else:
                raise ValueError('No file with the pretrained model selected')

            self.teacher_model.load_state_dict(checkpoint)

            self.student_model = EfficientNet.EfficientNet(config.student_efficient_net['input_features'],
                                                           config.student_efficient_net['model'])

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

        self.log_num = 1000
        self.num_epochs = utils.calculate_epochs(self.dataset, config.downsample['action'], self.M)
        self.counter = 0

        self.student_optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.1, momentum=0.9,
                                                 weight_decay=5e-4, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.student_optimizer, milestones=[0.3 * self.num_epochs - 1,
                                                                                            0.6 * self.num_epochs - 1,
                                                                                            0.8 * self.num_epochs - 1],
                                                        gamma=0.2)

    def train_KT_AT(self):
        # summary for current training loop and a running average object for loss
        # Use tqdm for progress bar
        accuracy_dict = {}

        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            self.student_model.train()
            self.train()

            if epoch % self.log_num == 0:
                acc = self.test()
                accuracy_dict[epoch] = acc
                utils.log_accuracy("KD_AT.csv", accuracy_dict)
                print(f"\nAccuracy: {acc:05.3f}")
                self.save_model()

            self.scheduler.step()

        utils.plot_accuracy("KD_AT.csv")

    def train(self):       
        running_acc = running_loss = 0

        for batch_num, input in enumerate(self.trainloader):
            self.student_optimizer.zero_grad()

            # move to GPU if available
            train_batch, labels_batch = input
            train_batch, labels_batch = train_batch.to(self.device), labels_batch.to(self.device)

            # compute model output, fetch teacher/student output, and compute KD loss
            student_logits, *student_activations = self.student_model(train_batch)
            teacher_logits, *teacher_activations = self.teacher_model(train_batch)

            # teacher/student outputs: logits, attention1, attention2, attention3

            loss = utils.KL_AT_loss(teacher_logits, student_logits, student_activations, teacher_activations,
                                    labels_batch)
            loss.backward()

            running_loss += loss.data
            running_acc += utils.accuracy(student_logits.data, labels_batch)
            self.counter += 1

            # performs updates using calculated gradients
            self.student_optimizer.step()

    def test(self, test=False):
        """Runs the model on test data.

        Keyword Arguments:
            test {bool} -- If it is being used in Test Mode. If so it needs to load the model from memory (default: {False})

        Raises:
            ValueError: If the pretrained model loaded from memory in Test Mode is not found

        Returns:
            {int} -- Accuracy on test data
        """        
        if test == True:
            if os.path.exists(self.student_save_path):
                checkpoint = torch.load(self.student_save_path, map_location=self.device)
            else:
                raise ValueError('No file with the pretrained model selected')

            self.student_model.load_state_dict(checkpoint)
        self.student_model.eval()

        running_acc = 0
        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.device), label.to(self.device)

                student_logits, *student_activations = self.student_model(data)

                running_acc += utils.accuracy(student_logits.data, label)

        print(f"Test accuracy: {running_acc / len(self.testloader)}")
        return running_acc / len(self.testloader)

    def save_model(self):
        torch.save(self.student_model.state_dict(), self.student_save_path)
