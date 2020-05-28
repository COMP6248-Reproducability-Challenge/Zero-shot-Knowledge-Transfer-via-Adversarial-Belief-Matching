import dataloaders
import ResNet
import torch
import torch.nn as nn
import utils
from tqdm import tqdm
import config
import EfficientNet
import os

class No_teacher:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset = config.dataset
        self.M = config.downsample['value']

        self.train_loader, self.test_loader, self.validation_loader, self.num_classes = dataloaders.transform_data(self.dataset, 
                                                                    M= config.downsample['value'], down= config.downsample['action'])
        self.model_type = config.model_type

        if self.model_type == "rnn":
            self.model = ResNet.WideResNet(depth=config.teacher_rnn['depth'], num_classes=self.num_classes, widen_factor=config.teacher_rnn['widen_factor'], 
                    input_features=config.teacher_rnn['input_features'], output_features=config.teacher_rnn['output_features'], 
                    dropRate=config.teacher_rnn['dropRate'], strides=config.teacher_rnn['strides'])
            
            if config.downsample['action']:
                self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-down_sample{config.downsample['value']}-seed{config.seed}.pth"
            else:
                self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-seed{config.seed}.pth"
        
        elif self.model_type == "efficient_net":
            self.model = EfficientNet.EfficientNet(config.teacher_efficient_net['input_features'], config.teacher_efficient_net['model'])

            if config.downsample['action']:
                self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-efficient_net-down_sample{config.downsample['value']}-seed{config.seed}.pth"
            else:
                self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-efficient_net-seed{config.seed}.pth"
        
        else:
            raise ValueError('Invalid model type')

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        else:
            print(f"Using {self.device}!")
        
        self.model.to(self.device)
    
    def train(self):
        print(f"Training {self.model_type}")
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.loss_function = torch.nn.CrossEntropyLoss()

        num_epochs = utils.calculate_epochs(self.dataset, config.downsample['action'], self.M)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=[0.3*num_epochs - 1,0.6*num_epochs - 1,0.8*num_epochs - 1], gamma=0.2)
        save_epochs = [0.2*num_epochs, 0.4*num_epochs, 0.6*num_epochs, 0.8*num_epochs, 0.99*num_epochs]
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            running_acc = 0
            running_loss = 0

            if epoch in save_epochs:
                torch.save(self.model.state_dict(), self.save_path)
            
            self.model.train()
            
            with tqdm(self.train_loader, total=len(self.train_loader), desc='train', position=0, leave=True) as t:
                for curr_batch, batch in enumerate(self.train_loader):
                    self.optimiser.zero_grad()
                    
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                    logits, *_ = self.model(data)

                    loss = self.loss_function(logits, labels)
                    loss.backward()

                    running_loss += loss.data
                    running_acc += utils.accuracy(logits, labels)

                    t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)), loss='{:05.3f}'.format(running_loss/(curr_batch+1)))
                    t.update()

                    self.optimiser.step()
            
            scheduler.step()

            if self.validation_loader is not None:
                self.validation()
        
        torch.save(self.model.state_dict(), self.save_path)
    
    def validation(self):
        self.model.eval()

        running_acc = 0
        running_loss = 0

        with torch.no_grad():
            with tqdm(self.validation_loader, total=len(self.validation_loader), desc='validation', position=0, leave=True) as t:
                for curr_batch, batch in enumerate(self.validation_loader):
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)

                    logits, *_ = self.model(data)
                    loss = self.loss_function(logits, labels)

                    running_loss += loss.data
                    running_acc += utils.accuracy(logits, labels)

                    t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)), loss='{:05.3f}'.format(running_loss/(curr_batch+1)))
                    t.update()
        
    def test(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=self.device)
        else:
            raise ValueError('No file with the pretrained model selected')

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        running_acc = 0

        with torch.no_grad():
            with tqdm(self.test_loader, total=len(self.test_loader), desc='test', position=0, leave=True) as t:
                for curr_batch, batch in enumerate(self.test_loader):
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)

                    logits, *_ = self.model(data)

                    running_acc += utils.accuracy(logits, labels)

                    t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)))
                    t.update()
