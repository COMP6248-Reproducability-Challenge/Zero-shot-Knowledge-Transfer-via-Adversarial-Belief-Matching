import dataloaders
import ResNet
import torch
import utils
from tqdm import tqdm
import config

class No_teacher:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset = config.dataset
        self.train_loader, self.test_loader, self.validation_loader, self.num_classes = dataloaders.transform_data(self.dataset, 
                                                                    M= config.downsample['value'], down= config.downsample['action'])

        self.model = ResNet.WideResNet(depth=config.teacher_rnn['depth'], num_classes=self.num_classes, widen_factor=config.teacher_rnn['widen_factor'], 
                    input_features=config.teacher_rnn['input_features'], output_features=config.teacher_rnn['output_features'], 
                    dropRate=config.teacher_rnn['dropRate'], strides=config.teacher_rnn['strides'])
        self.model.to(self.device)

        if config.downsample['action']:
            self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-down_sample{config.downsample['value']}-seed{config.seed}.pth"
        else:
            self.save_path = f"{config.save_path}/{self.dataset}-{config.mode}-wrn-{config.teacher_rnn['depth']}-{config.teacher_rnn['widen_factor']}-{config.teacher_rnn['dropRate']}-seed{config.seed}.pth"
    
    def train(self):
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.loss_function = torch.nn.CrossEntropyLoss()

        if self.dataset == "cifar10":
            num_epochs= 200
        else:
            num_epochs= 100

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=[0.3*num_epochs - 1,0.6*num_epochs - 1,0.8*num_epochs - 1], gamma=0.2)
        save_epochs = [50,99,150,199]
        
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
                    logits, _, _, _ = self.model(data)

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
        self.test()
    
    def validation(self):
        self.model.eval()

        running_acc = 0
        running_loss = 0

        with torch.no_grad():
            with tqdm(self.validation_loader, total=len(self.validation_loader), desc='validation', position=0, leave=True) as t:
                for curr_batch, batch in enumerate(self.validation_loader):
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)

                    logits, _, _, _ = self.model(data)
                    loss = self.loss_function(logits, labels)

                    running_loss += loss.data
                    running_acc += utils.accuracy(logits, labels)

                    t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)), loss='{:05.3f}'.format(running_loss/(curr_batch+1)))
                    t.update()
        
    def test(self):
        self.model.eval()

        running_acc = 0

        with torch.no_grad():
            with tqdm(self.test_loader, total=len(self.test_loader), desc='test', position=0, leave=True) as t:
                for curr_batch, batch in enumerate(self.test_loader):
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)

                    logits, _, _, _ = self.model(data)

                    running_acc += utils.accuracy(logits, labels)

                    t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)))
                    t.update()
