from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import config
import ResNet
writer = SummaryWriter()


def setup_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_teacher_rnn():
    teacher = ResNet.WideResNet(depth=config.teacher_rnn['depth'], num_classes=self.num_classes,
                                               widen_factor=config.teacher_rnn['widen_factor'],
                                               input_features=config.teacher_rnn['input_features'],
                                               output_features=config.teacher_rnn['output_features'],
                                               dropRate=config.teacher_rnn['dropRate'],
                                               strides=config.teacher_rnn['strides'])
    
    return teacher

def load_student_rnn():
    student = ResNet.WideResNet(depth=config.student_rnn['depth'], num_classes=self.num_classes,
                                               widen_factor=config.student_rnn['widen_factor'],
                                               input_features=config.student_rnn['input_features'],
                                               output_features=config.student_rnn['output_features'],
                                               dropRate=config.student_rnn['dropRate'],
                                               strides=config.student_rnn['strides'])
    
    return student

def calculate_epochs(dataset, downsample, downsamplevalue):
    if dataset == "cifar10":
        num_epochs = 200
        if downsample:
            if downsamplevalue == 0:
                num_epochs = 0
            else:
                num_epochs = int(num_epochs * 50000 / (10 * downsamplevalue))
    elif dataset == "svhn":
        num_epochs = 100
        if downsample:
            if downsamplevalue == 0:
                num_epochs = 0
            else:
                num_epochs = int(num_epochs * 73257 / (10 * downsamplevalue))
    else:
        num_epochs= 170
        if downsample:
            if downsamplevalue == 0:
                num_epochs = 0
            else:
                num_epochs = int(num_epochs * 60000 / (10 * downsamplevalue))

    return num_epochs


def KL_AT_loss(student_logits, teacher_logits, student_activations, teacher_activations, labels,
               temperature=1.0, alpha=0.9, beta=1000):
    kl_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                       F.softmax(teacher_logits / temperature, dim=1))  # forward KL
    kl_loss *= (temperature ** 2) * 2

    cross_entropy = F.cross_entropy(student_logits, labels)

    attention_loss = 0
    for x in range(len(student_activations)):
        attention_loss += attention_diff(student_activations[x], teacher_activations[x])

    adjusted_beta = (beta * 3) / len(student_activations)
    attention_loss *= adjusted_beta
    # beta value taken directly from other code, no explanation given

    loss = (1.0 - alpha) * cross_entropy + kl_loss * alpha + attention_loss

    return loss


def KL_Loss(student_logits, teacher_logits, temperature=1.0):
    kl_loss = torch.nn.KLDivLoss()(F.log_softmax(student_logits / temperature, dim=1),
                                   F.softmax(teacher_logits / temperature, dim=1))

    return kl_loss


def student_loss_zero_shot(student_outputs, teacher_outputs, b=250):
    """
    Taken from hhttps://github.com/AlexandrosFerles/NIPS_2019_Reproducibilty_Challenge_Zero-shot_Knowledge_Transfer_via_Adversarial_Belief_Matching
    """

    student_out, student_activations = student_outputs[0], student_outputs[1:]
    teacher_out, teacher_activations = teacher_outputs[0], teacher_outputs[1:]

    activation_pairs = zip(student_activations, teacher_activations)

    attention_losses = [attention_diff(att1, att2) for (att1, att2) in activation_pairs]
    loss_term1 = b * sum(attention_losses)
    loss = loss_term1 - (-KL_Loss(student_out, teacher_out))

    return loss


def attention(x):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    """
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    """
    return (attention(x) - attention(y)).pow(2).mean()


def accuracy(logits, data):
    _, predictions = torch.max(logits, 1)
    total = data.size(0)
    correct = (predictions == data).sum().item()

    return correct / total


def log_accuracy(logfile_name, accuracy_dict):
    Path("./logs/").mkdir(parents=True, exist_ok=True)

    logfile_name = "./logs/" + (logfile_name if "." in logfile_name else logfile_name + ".csv")
    f = open(logfile_name, "w+")
    f.write("Epoch,Accuracy\n")

    for key, value in accuracy_dict.items():
        f.write(str(key) + "," + str(value) + "\n")

    f.close()


def plot_accuracy(logfile_name, save_plot=True):
    logfile_name = logfile_name if "." in logfile_name else logfile_name + ".csv"
    data = np.genfromtxt(f'./logs/{logfile_name}', delimiter=',', skip_header=1,
                         names=['Epochs', 'Accuracy'])
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.set_title("Accuracy plot")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(data['Epochs'], data['Accuracy'], color='r', label='Accuracy per epoch')
    leg = ax1.legend()
    # plt.show()

    if save_plot:
        Path("./plots/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f'./plots/{logfile_name.split(".")[0]}.png')


def writeMetrics(value_dict, step):
    for key, value in value_dict.items():
        writer.add_scalar(key, value, step)
