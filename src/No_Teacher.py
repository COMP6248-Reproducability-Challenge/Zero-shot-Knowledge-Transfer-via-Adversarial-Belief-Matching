import dataloaders
import ResNet
import torch
from torchbearer import Trial
from torchbearer import callbacks
import utils

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
                        output_features=output_features, dropRate=dropRate, strides=strides, noTeacher=True)
    torch_checkpoint = torch.load('../PreTrainedModels/cifar10-no_teacher-wrn-40-2-0.0-seed0.pth', map_location=device)
    model.load_state_dict(torch_checkpoint)
    model.train()

    optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    metrics = ['loss', 'accuracy']

    if dataset == "cifar10":
        num_epochs= 200
    else:
        num_epochs= 100
    
    scheduler = callbacks.torch_scheduler.MultiStepLR(milestones=[0.3*num_epochs,0.6*num_epochs,0.8*num_epochs], gamma=0.2)
    full_path = save_path + "/" + dataset + f"-no_teacher-wrn-{depth}-{widen_factor}-{dropRate}-seed{seed}.pth"
    checkpoint = callbacks.Interval(full_path, period=50, on_batch=True, save_model_params_only=True)

    trial = Trial(model, optimiser, loss_function, metrics=metrics, callbacks=[scheduler, checkpoint]).to(device)

    if validation_loader is None:
        trial.with_generators(train_loader, test_generator=test_loader)
    else:
        trial.with_generators(train_loader, val_generator=validation_loader, test_generator=test_loader)

    # trial.run(epochs=num_epochs)
    # state_dict = trial.state_dict()["model"]
    # torch.save(state_dict, full_path)
    model.eval()
    predictions = trial.predict()
