import dataloaders
import ResNet
import torch
from torchbearer import Trial
import utils

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def No_teacher(dataset):
    train_loader, test_loader, validation_loader, num_classes = dataloaders.transform_data(dataset)
    
    model = ResNet.WideResNet(depth= 40, num_classes= num_classes, widen_factor= 2, dropRate= 0.0, noTeacher= True)

    optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    metrics = ['loss', 'accuracy']
    num_epochs= 200

    trial = Trial(model, optimiser, loss_function, metrics=metrics).to(device)

    if validation_loader is None:
        trial.with_generators(train_loader, test_generator=test_loader)
    else:
        trial.with_generators(train_loader, val_generator=validation_loader, test_generator=test_loader)

    trial.run(epochs=num_epochs)
    predictions = trial.predict()
    print("Accuracy: " + str(utils.accuracy()))
