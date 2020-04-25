import torch.nn.functional as F




def KL_loss(student_logits, teacher_logits, temperature =1.0):
    """
       Taken from https://github.com/polo5/ZeroShotKnowledgeTransfer
    """




    kl_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                          F.softmax(teacher_logits / temperature, dim=1))  # forward KL



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