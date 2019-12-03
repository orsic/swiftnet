import torch.nn.functional as F

__all__ = ['cross_entropy_with_logits', 'cross_entropy_with_logits_and_hist', 'mean_squared_error']


def cross_entropy_with_logits(y, t):
    '''
    :param y: Tensor of logits
    :param t: Tensor of logits
    :return:
    '''
    assert y.shape == t.shape
    return -(y.log_softmax(dim=1) * t.softmax(dim=1)).sum(dim=1).mean()


def cross_entropy_with_logits_and_hist(y, t, reduce=True):
    '''
    :param y: Tensor of logits
    :param t: Tensor of histograms
    :return:
    '''
    assert y.shape == t.shape
    ce = -(y.log_softmax(dim=1) * t).sum(dim=1)
    if reduce:
        ce = ce.mean()
    return ce


def mean_squared_error(y, t):
    '''
    :param y: Tensor of logits
    :param t: Tensor of logits
    :return:
    '''
    assert y.shape == t.shape
    return F.mse_loss(y, t, reduction='mean')
