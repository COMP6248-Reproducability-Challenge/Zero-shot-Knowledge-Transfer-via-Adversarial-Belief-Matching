import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
import torchvision.transforms as transforms
import torchvision

def setup_seeds():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

def KL_AT_loss(student_logits, teacher_logits,student_activations, teacher_activations,labels,
               temperature =1.0, alpha=0.9, beta=1000):

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


def KL_Loss(student_logits, teacher_logits, temperature =1.0):

    kl_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                          F.softmax(teacher_logits / temperature, dim=1))

    return kl_loss


def attention(x):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    """
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()

def accuracy(model, data, device):
    model.eval()
    labels = []
    predictions = []

    for img, label in data:
        with torch.no_grad():
            logits = model(img.to(device))[0]

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
        labels.append(label)

    predictions = np.array(predictions)
    labels = np.array([x.numpy() for x in labels])
    predictions = np.argmax(predictions, axis=1).flatten()
    labels = labels.flatten()
    size = len(labels)

    return np.sum(predictions == labels) / size

def checkpoint(model, path):
    torch.save(model.state_dict(), path)
