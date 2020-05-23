import dataloaders
import ResNet
import torch
import utils
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def No_teacher(save_path, dataset, seed):
    train_loader, test_loader, validation_loader, num_classes = dataloaders.transform_data(dataset)
    
    strides = [1, 2, 2]
    depth = 40
    widen_factor = 2
    dropRate = 0.0
    input_features = 3
    output_features = 16

    model = ResNet.WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, input_features=input_features,
                        output_features=output_features, dropRate=dropRate, strides=strides)
    model.to(device)

    optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    if dataset == "cifar10":
        num_epochs= 200
    else:
        num_epochs= 100

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[0.3*num_epochs,0.6*num_epochs,0.8*num_epochs], gamma=0.2)
    save_epochs = [50,99,150,199]
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        running_acc = 0
        running_loss = 0

        if epoch in save_epochs:
            utils.checkpoint(save_path, dataset, model, "no_teacher", depth, widen_factor, dropRate, seed)  

        model.train()
         
        with tqdm(train_loader, total=len(train_loader), desc='train', position=0, leave=True) as t:
            for curr_batch, batch in enumerate(train_loader):
                optimiser.zero_grad()
                
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                logits, _, _, _ = model(data)

                loss = loss_function(logits, labels)
                loss.backward()

                running_loss += loss.data
                running_acc += utils.accuracy(logits, labels)

                t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)), loss='{:05.3f}'.format(running_loss/(curr_batch+1)))
                t.update()

                optimiser.step()
        
        scheduler.step()

        if validation_loader is not None:
            validation(model, validation_loader)
    
    utils.checkpoint(save_path, dataset, model, "no_teacher", depth, widen_factor, dropRate, seed)
    test(model, test_loader)
    
def validation(model, validation_loader):
    model.eval()

    running_acc = 0
    running_loss = 0

    with torch.no_grad():
        with tqdm(validation_loader, total=len(validation_loader), desc='validation', position=0, leave=True) as t:
            for curr_batch, batch in enumerate(validation_loader):
                data, labels = batch
                data, labels = data.to(device), labels.to(device)

                logits, _, _, _ = model(data)
                loss = loss_function(logits, labels)

                running_loss += loss.data
                running_acc += utils.accuracy(logits, labels)

                t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)), loss='{:05.3f}'.format(running_loss/(curr_batch+1)))
                t.update()
    
def test(model, test_loader):
    model.eval()

    running_acc = 0

    with torch.no_grad():
        with tqdm(test_loader, total=len(test_loader), desc='test', position=0, leave=True) as t:
            for curr_batch, batch in enumerate(test_loader):
                data, labels = batch
                data, labels = data.to(device), labels.to(device)

                logits, _, _, _ = model(data)

                running_acc += utils.accuracy(logits, labels)

                t.set_postfix(accuracy='{:05.3f}'.format(running_acc/(curr_batch+1)))
                t.update()
