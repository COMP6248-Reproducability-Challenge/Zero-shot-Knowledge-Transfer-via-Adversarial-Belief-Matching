import torch.nn.functional as F




def KL_AT_loss(student_logits, teacher_logits,student_activations, teacher_activations,labels,
               temperature =1.0, alpha=0.9, beta=1000):

    kl_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                          F.softmax(teacher_logits / temperature, dim=1))  # forward KL
    kl_loss *= (temperature ** 2) * 2

    cross_entropy = F.cross_entropy(student_logits, labels)

    attention_loss = 0
    for x in range(student_activations):
        attention_loss += attention_diff(student_activations[x], teacher_activations[x])

    adjusted_beta = (beta * 3) / len(student_activations)
    attention_loss *= adjusted_beta
    # beta value taken directly from other code, no explanation given

    loss = (1.0 - alpha) * cross_entropy + kl_loss * alpha + attention_loss

    return loss

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