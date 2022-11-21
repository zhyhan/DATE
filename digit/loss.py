import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#from grl import WarmStartGradientReverseLayer, GradientReverseLayer


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is
    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].
    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.
    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None):
        super(DomainAdversarialLoss, self).__init__()
        #self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=100, auto_step=True) if grl is None else grl
        self.grl = GradientReverseLayer()
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, feature: torch.Tensor, labels: torch.Tensor, weights: Optional = None) -> torch.Tensor:
        f = self.grl(feature)
        d = self.domain_discriminator(f)

        #d_s, d_t = d.chunk(2, dim=0)
        #d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        #d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        labels = labels.float()
        self.domain_discriminator_accuracy = binary_accuracy(d, labels.float())
        #print(labels,labels.size(), labels.type())
        w_s = weights
        #w_s = torch.ones_like(labels)
        #weights_test =  d.mean(dim=0)
        # if w_s is None:
        #     w_s = torch.ones_like(d_label_s)
        # if w_t is None:
        #     w_t = torch.ones_like(d_label_t)
        return self.domain_discriminator_accuracy, self.bce(d, labels.view_as(d), w_s.view_as(d))

def KLConsistencyLoss(output, pred_label, args, temperature=2):
    """
    Class-Relation-Aware Consistency Loss
    Args:
        output: n x b x k (source num x batch size x class num)
        pred_label:  b x 1
        args:   argments
    """
    eps = 1e-16
    KL_loss = 0

    label_id = pred_label.cpu().numpy()
    label_id = np.unique(label_id)

    for cls in range(args.class_num):
        if cls in label_id:
            prob_cls_all = torch.ones(len(args.src), args.class_num)

            for i in range(len(args.src)):
                mask_cls =  pred_label.cpu() == cls
                mask_cls_ex = torch.repeat_interleave(mask_cls.unsqueeze(1), args.class_num, dim=1)

                logits_cls = torch.sum(output[i] * mask_cls_ex.float(), dim=0)
                cls_num = torch.sum(mask_cls)
                logits_cls_acti = logits_cls * 1.0 / (cls_num + eps)
                prob_cls = torch.softmax(logits_cls_acti, dim=0)
                prob_cls = torch.clamp(prob_cls, 1e-8, 1.0)

                prob_cls_all[i] = prob_cls


            for m in range(len(args.src)):
                for n in range(len(args.src)):
                    KL_div = torch.sum(prob_cls_all[m] * torch.log(prob_cls_all[m] / prob_cls_all[n])) + \
                              torch.sum(prob_cls_all[n] * torch.log(prob_cls_all[n] / prob_cls_all[m]))
                    KL_loss += KL_div / 2

    KL_loss = KL_loss / (args.class_num * len(args.src))

    return KL_loss

