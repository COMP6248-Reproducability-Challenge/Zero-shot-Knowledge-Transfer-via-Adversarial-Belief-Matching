import dataloaders
import ResNet
from torch import nn
from torch import optim

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def No_teacher():
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model_teacher.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    metrics = ['loss', 'accuracy']
    num_epochs= 200

    train_loader, test_loader, validation_loader, num_classes = dataloaders.transform_data("CIFAR10")
    model = ResNet.WideResNet(depth= 40, num_classes= num_classes, widen_factor= 2, dropRate= 0.0, noTeacher= True)

    trial = Trial(model, optimiser, loss_function, metrics=metrics).to(device)

    if validation_loader is None:
        trial.with_generators(train_loader, test_generator=test_loader)
    else:
        trial.with_generators(train_loader, val_generator=validation_loader, test_generator=test_loader)

    trial.run(epochs=num_epochs)
    predictions = trail.predict()
    print("Accuracy: " + str(utils.accuracy()))