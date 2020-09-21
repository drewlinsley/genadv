import os
# import ipdb
import torch
import numpy as np
from torch import nn, optim
from torchvision import models
import torch.nn.functional as F
from models import siamese_resnet
from models import alexnet
from models import vgg
from models import mnasnet
from models import inception
from models import densenet
from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.models.model_store import download_model
from matplotlib import pyplot as plt
from resources.pytorch_pretrained_biggan import one_hot_from_names
from collections import OrderedDict

from torch.distributions.kl import kl_divergence
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal  # noqa
from torch.distributions.normal import Normal
from torch.distributions.half_normal import HalfNormal
from torch.distributions.bernoulli import Bernoulli
# from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
# from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


CLASSES = [
    'border collie',
    'arctic fox',
    'penguin',
    'toucan',
    'gorilla',
    'tabby',
    'fly',
    'goldfish',
    'airplane',
    'sports car',
    'sunglasses',
    'hotdog',
    'speedboat',
    'missile',
    'desk',
    'church'
]


def plot_fun(path, n, idx, P, ext='.png', rows=4):
    """Plot some images and save to path.

    Requires

    path <str> directory for saving
    n <int> number of images to plot
    P <object> dataset generator
    ext <str> image extension
    rows <int> number of rows for plotting
    """
    if 'gan' in P.dataset:
        class_vector = one_hot_from_names(CLASSES)
        assert class_vector is not None
        with torch.no_grad():
            batch, labels = P.sample_batch(
                len(class_vector), class_vector=class_vector)
    else:
        with torch.no_grad():
            batch, labels = P.sample_batch(n)
    if P.siamese:
        batch = batch.sum(-1)  # Siamese
    batch = batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    norm_mean = P.norm_mean.cpu().numpy().reshape(1, 1, 3)
    norm_std = P.norm_std.cpu().numpy().reshape(1, 1, 3)
    batch = batch * norm_std + norm_mean
    columns = n / rows
    f = plt.figure()
    for xx in range(n):
        plt.subplot(rows, columns, xx + 1)
        plt.imshow(batch[xx])
        plt.axis('off')
        plt.title('Label: {}'.format(labels[xx]))
    plt.savefig('{}_{}{}'.format(path, idx, ext))
    plt.close(f)


def plot_examples(path, n_subplots, n_batches, P, ext='.png', rows=4):
    """Plot some images and save to path.

    Requires

    path <str> directory for saving
    n <int> number of images to plot
    P <object> dataset generator
    ext <str> image extension
    rows <int> number of rows for plotting
    """
    for idx in range(n_batches):
        plot_fun(
            path=path,
            n=n_subplots,
            idx=idx,
            P=P,
            ext=ext,
            rows=rows)


def get_examples(P, n=16, convert_to_numpy=True):
    """Postprocess a batch of images.

    TODO: Consolidate with plotting code above."""
    if 'gan' in P.dataset:
        class_vector = one_hot_from_names(CLASSES)
        assert class_vector is not None
        with torch.no_grad():
            batch, labels = P.sample_batch(
                len(class_vector), class_vector=class_vector)
    else:
        with torch.no_grad():
            batch, labels = P.sample_batch(n)
    if P.siamese:
        batch = batch.sum(-1)  # Siamese
    if convert_to_numpy:
        batch = batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
        norm_mean = P.norm_mean.cpu().numpy().reshape(1, 1, 3)
        norm_std = P.norm_std.cpu().numpy().reshape(1, 1, 3)
    else:
        batch = batch.detach().cpu().permute(0, 2, 3, 1)
        norm_mean = P.norm_mean.cpu().reshape(1, 1, 3)
        norm_std = P.norm_std.cpu().reshape(1, 1, 3)
    batch = batch * norm_std + norm_mean
    return batch


def get_optimizer(net, lr, optimizer, trainable, amsgrad=True, momentum=0.9):
    """Create optimizer for a network."""
    params_to_update = net.parameters()
    print("Params to learn:")
    if not trainable:
        params_to_update = []
        for name, param in net.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print(('{:10} {}'.format('', name)))
    else:
        for name, param in net.named_parameters():
            if param.requires_grad is True:
                print(('{:10} {}'.format('', name)))
    opt = getattr(optim, optimizer)
    if 'adam' in optimizer.lower():
        return opt(params_to_update, lr=lr, amsgrad=amsgrad)
    elif 'sgd' in optimizer.lower():
        return opt(params_to_update, lr=lr, momentum=momentum)
    else:
        raise NotImplementedError(optimizer)


def set_parameter_requires_grad(model, trainable=False):
    """Adjust grad dependencies for a model."""
    if trainable:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False


def adapt_beta(loss, history, beta, scale=0.5, cutoff=1000, threshold=0.2):
    """Scale beta if nothing has happened in cutoff steps."""
    if len(history) < cutoff:
        return beta
    rho = np.abs(np.corrcoef(history, np.arange(len(history)))[0, 1])
    if rho < threshold:
        print('Beta history slope: {}. Scaling from {} to {}.'.format(rho, beta, beta * scale))  # noqa
        return beta * scale
    return beta


def initialize_model(
        dataset,
        model_name,
        num_classes,
        pretrained,
        trainable=True,
        siamese_version=None,
        siamese=False):
    """Initialize these variables which will be set in this if
    statement. Each of these variables is model specific."""
    model_ft = None
    input_size = 0
    if 'resnet' in model_name:
        model_fun = getattr(siamese_resnet, model_name)
        model_ft = model_fun(
            pretrained=pretrained, siamese=siamese, version=siamese_version)
        set_parameter_requires_grad(model_ft, trainable)
        if 'gan' not in dataset:
            if model_ft.version == 'conv':
                num_ftrs = model_ft.fc.in_channels
                model_ft.fc = nn.Conv1d(
                    num_ftrs, num_classes, kernel_size=2, bias=True, stride=2)
                nn.init.kaiming_normal_(
                    model_ft.fc.weight, mode='fan_out', nonlinearity='linear')
            else:
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'cbam':
        model_ft = ptcv_get_model(
            'cbam_resnet50', pretrained=pretrained)  # Fix this API
        if pretrained:
            root = os.path.join('~', '.torch', 'models')
            print('WARNING: Using CBAM Pretrained. Needs testing.')
            setattr(model_ft, '_initialize', lambda: download_model(
                net=model_ft,
                model_name='cbam_resnet50', local_model_store_dir_path=root))
            # setattr(model_ft, '_initialize', lambda: 0 + 0)
        else:
            setattr(model_ft, '_initialize', model_ft._init_params)
        num_ftrs = model_ft.output.in_features
        if 'gan' not in dataset:
            model_ft.output = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = alexnet.alexnet(
            pretrained=pretrained, siamese=siamese, version=siamese_version)
        set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.classifier[6].in_features
        if 'gan' not in dataset:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'vgg' in model_name:
        model_fun = getattr(vgg, model_name)
        model_ft = model_fun(
            pretrained=pretrained, siamese=siamese, version=siamese_version)
        set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.classifier[6].in_features
        if 'gan' not in dataset:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif "mnasnet" in model_name:
        model_fun = getattr(mnasnet, model_name)
        model_ft = model_fun(
            pretrained=pretrained, siamese=siamese, version=siamese_version)
        set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.classifier[-1].in_features
        if 'gan' not in dataset:
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        raise NotImplementedError
        model_ft = models.squeezenet1_0(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, trainable)
        model_ft.classifier[1] = nn.Conv2d(
            512,
            num_classes,
            kernel_size=(1, 1),
            stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif 'densenet' in model_name:
        model_fun = getattr(densenet, model_name)
        model_ft = model_fun(
            pretrained=pretrained,
            siamese=siamese,
            version=siamese_version)
        set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.classifier.in_features
        if 'gan' not in dataset:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'inception' in model_name:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_fun = getattr(inception, model_name)
        model_ft = model_fun(
            pretrained=pretrained,
            siamese=siamese,
            version=siamese_version)
        set_parameter_requires_grad(model_ft, trainable)
        if 'gan' not in dataset:
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    else:
        raise NotImplementedError(model_name)
    return model_ft, input_size


def weight_norm_op(v, g):
    """Weight normalization."""
    return v * (g / np.linalg.norm(v, ord='fro')).expand_as(v)


def normalize_weights(P, name, prop):
    """Apply weight normalization."""
    g_attr_name = '{}_{}'.format(name, '{}_g'.format(prop))
    v_attr_name = '{}_{}'.format(name, '{}_v'.format(prop))
    if 'Npz' in str(type(P)) or isinstance(P, dict):
        g = P.get(g_attr_name, None)
        v = P.get(v_attr_name, None)
    else:
        g = getattr(P, g_attr_name)
        v = getattr(P, v_attr_name)
    try:
        return v * (g / torch.norm(v)).expand_as(v)
    except Exception:
        return v


def get_z(dists, P, eps=1e-8):
    """Return L2 on the z scores of a normal distribution."""
    l2 = 0.
    for dst in dists:
        name = dst['name']
        family = dst['family']
        if family == 'normal' or family == 'abs_normal' or family == 'gaussian':
            if P.wn:
                lambda_r_mu = normalize_weights(P=P, name=name, prop='center')
                lambda_r_scale = normalize_weights(P=P, name=name, prop='scale')  # noqa
            else:
                attr_name = '{}_{}'.format(name, 'center')
                lambda_r_mu = getattr(P, attr_name)
                attr_name = '{}_{}'.format(name, 'scale')
                lambda_r_scale = getattr(P, attr_name)
            lambda_0_mu = dst['lambda_0']
            lambda_0_scale = dst['lambda_0_scale']
            z1 = lambda_r_mu / (eps + torch.std(lambda_r_scale))
            z0 = lambda_0_mu / (eps + torch.std(lambda_0_scale))
            score = torch.mean((z1 - z0) ** 2)
        else:
            raise NotImplementedError(family)
        if torch.isnan(score):  # Give a numerical margin
            raise RuntimeError('{} hist a nan.'.format(name))
        l2 = l2 + score
    return l2


def get_kl(dists, P, eps=1e-4, eta=1e-20, Fdiv='kl'):
    """Get KL divergences for different distributions."""
    kl = 0.
    for dst in dists:
        name = dst['name']
        family = dst['family']
        if family == 'gaussian' or family == 'mv_normal':
            if P.wn:
                lambda_r_mu = normalize_weights(P=P, name=name, prop='center')
                lambda_r_scale = normalize_weights(P=P, name=name, prop='scale')  # noqa
            else:
                attr_name = '{}_{}'.format(name, 'center')
                lambda_r_mu = getattr(P, attr_name)
                attr_name = '{}_{}'.format(name, 'scale')
                lambda_r_scale = getattr(P, attr_name)
            lambda_0_mu = dst['lambda_0']
            lambda_0_scale = dst['lambda_0_scale']
            lambda_r_dist = MultivariateNormal(loc=lambda_r_mu, covariance_matrix=lambda_r_scale)  # noqa
            lambda_0_dist = MultivariateNormal(loc=lambda_0_mu, covariance_matrix=lambda_0_scale)  # noqa
        elif family == 'low_mv_normal':
            if P.wn:
                lambda_r_mu = normalize_weights(P=P, name=name, prop='center')
                lambda_r_scale = normalize_weights(P=P, name=name, prop='scale')  # noqa
                lambda_r_factor = normalize_weights(P=P, name=name, prop='factor')  # noqa
            else:
                attr_name = '{}_{}'.format(name, 'center')
                lambda_r_mu = getattr(P, attr_name)
                attr_name = '{}_{}'.format(name, 'scale')
                lambda_r_scale = getattr(P, attr_name)
                attr_name = '{}_{}'.format(name, 'factor')
                lambda_r_factor = getattr(P, attr_name)
            lambda_0_mu = dst['lambda_0']
            lambda_0_scale = dst['lambda_0_scale']
            lambda_0_factor = dst['lambda_0_factor']
            lambda_r_dist = LowRankMultivariateNormal(loc=lambda_r_mu, cov_diag=lambda_r_scale, cov_factor=lambda_r_factor)  # noqa
            lambda_0_dist = LowRankMultivariateNormal(loc=lambda_0_mu, cov_diag=lambda_0_scale, cov_factor=lambda_0_factor)  # noqa
        elif family == 'normal' or family == 'abs_normal' or family == 'cnormal':
            if P.wn:
                lambda_r_mu = normalize_weights(P=P, name=name, prop='center')
                lambda_r_scale = normalize_weights(P=P, name=name, prop='scale')  # noqa
            else:
                attr_name = '{}_{}'.format(name, 'center')
                lambda_r_mu = getattr(P, attr_name)
                attr_name = '{}_{}'.format(name, 'scale')
                lambda_r_scale = getattr(P, attr_name)
            lambda_0_mu = dst['lambda_0']
            lambda_0_scale = dst['lambda_0_scale']
            lambda_r_dist = Normal(loc=lambda_r_mu, scale=lambda_r_scale)
            lambda_0_dist = Normal(loc=lambda_0_mu, scale=lambda_0_scale)
        elif family == 'half_normal':
            if P.wn:
                lambda_r_scale = normalize_weights(P=P, name=name, prop='scale')  # noqa
            attr_name = '{}_{}'.format(name, 'scale')
            lambda_r_scale = getattr(P, attr_name)
            lambda_0_scale = dst['lambda_0_scale']
            lambda_r_dist = HalfNormal(scale=lambda_r_scale)
            lambda_0_dist = HalfNormal(scale=lambda_0_scale)
        elif family == 'categorical':
            if P.wn:
                lambda_r_mu = normalize_weights(P=P, name=name, prop='center')
            attr_name = '{}_{}'.format(name, 'center')
            lambda_r_mu = getattr(P, attr_name)
            lambda_0 = dst['lambda_0']  # This is probs
            log_0 = (lambda_0 + eta).log()
            # noqa lambda_r_dist = RelaxedOneHotCategorical(temperature=1e-1, logits=lambda_r_mu)  # Log probs
            # noqa lambda_0_dist = RelaxedOneHotCategorical(temperature=1e-1, logits=log_0)
            lambda_r_dist = OneHotCategorical(logits=lambda_r_mu)  # Log probs
            lambda_0_dist = OneHotCategorical(logits=log_0)
        elif family == 'relaxed_bernoulli':
            if P.wn:
                lambda_r_mu = normalize_weights(P=P, name=name, prop='center')
            attr_name = '{}_{}'.format(name, 'center')
            lambda_r_mu = getattr(P, attr_name)
            lambda_0 = dst['lambda_0']  # This is probs
            log_0 = (lambda_0 + eta).log()
            # noqa lambda_r_dist = RelaxedBernoulli(temperature=1e-1, logits=lambda_r_mu)  # Log probs
            # noqa lambda_0_dist = RelaxedBernoulli(temperature=1e-1, logits=log_0)
            lambda_r_dist = Bernoulli(logits=lambda_r_mu)  # Log probs
            lambda_0_dist = Bernoulli(logits=log_0)
        else:
            raise NotImplementedError(
                'KL for {} is not implemented.'.format(family))
        if Fdiv == 'kl':
            it_kl = kl_divergence(p=lambda_0_dist, q=lambda_r_dist).sum()
        elif Fdiv == 'js':
            raise RuntimeError('Needs per-distribution implementation.')
            m = 0.5 * (lambda_0_dist.probs * lambda_r_dist.probs)
            p = kl_divergence(p=lambda_0_dist, q=m).sum()
            q = kl_divergence(p=lambda_r_dist, q=m).sum()
            it_kl = 0.5 * p + 0.5 * q
        else:
            raise NotImplementedError(div)
        if it_kl < -1e-4 or torch.isnan(it_kl):  # Give a numerical margin
            print(kl)
        kl = kl + it_kl
    return kl


def normalize_fun(tensor, mean, std, max_val=255., reshape=(1, 3, 1, 1)):
    """Apply zscore normalization."""
    if reshape is not None:
        tensor = tensor.unsqueeze(1).repeat(reshape)
    else:
        reshape = (1, 3, 1, 1)
    tensor = tensor / max_val
    mean = mean.view(reshape)
    std = std.view(reshape)
    return (tensor - mean) / std


def sample_gumbel(logits, eps=1e-20):
    """Sample from rand distribution and convert to logit."""
    U = torch.rand_like(logits)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """Get softmax distribution.
    Note that temperature can be fixed or learned."""
    y = logits + sample_gumbel(logits)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(param, temperature=1e-1, hard=True, num_samples=1):
    """
    Straight-through estimator (detatch) to allow for argmax.
    input: [*, n_class] param, which is in log space
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits=param, temperature=temperature)
    if hard:
        # One-hot the y
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).scatter_(-1, ind, 1.0)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def prep_params(params):
    """Prepare parameters for saving."""
    if hasattr(params, 'named_parameters'):
        return {
            k: v.detach().cpu().numpy().tolist()
            for k, v
            in params.named_parameters()}
    else:
        return {
            k: v.detach().cpu().numpy().tolist()
            for k, v
            in params.items()}


def reversal(grad):
    """Reverse direction of gradients."""
    grad_clone = grad.clone()
    return grad_clone.neg()


def set_model_trainable(net):
    """Set all param grads to trainable."""
    for param in net.parameters():
        param.requires_grad = True


def set_model_not_trainable(net):
    """Set all param grads to not trainable."""
    for param in net.parameters():
        param.requires_grad = False


def set_lrs(optimizer, lr):
    """Set the learning rate of an optimizer."""
    for param in optimizer.param_groups:
            param['lr'] = lr


def inner_loop(
        net,
        net_loss,
        net_optimizer,
        P,
        device,
        inner_pbar,
        warm_up=False,
        batch_size=1):
    """Run model optim."""
    with torch.no_grad():
        batch, labels = P.sample_batch(batch_size)
    out = net(batch)
    L = net_loss(out, labels)
    if warm_up:
        desc = 'Warm-up loss is {:.4f}'.format(L)
        inner_pbar.set_description(desc)
        inner_pbar.update(1)
        return L
    L.backward()
    net_optimizer.step()
    net_optimizer.zero_grad()
    desc = 'Loss is {:.4f}'.format(L)
    inner_pbar.set_description(desc)
    inner_pbar.update(1)
    return L


def roll(
        x: torch.Tensor,
        shift: int,
        dim: int=-1,
        fill_pad: int=0.) -> torch.Tensor:
    """Roll a tensor along a dimension."""
    if 0 == shift:
        return x
    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift).to(x.device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat(
            [x.index_select(
                dim, torch.arange(shift, x.size(dim))), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(
            dim, torch.arange(shift, x.size(dim)).to(x.device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat(
            [gap, x.index_select(
                dim, torch.arange(shift).to(x.device))], dim=dim)


def outer_loop(
        batch_size,
        outer_batch_size_multiplier,
        adv_version,
        num_classes,
        net,
        net_loss,
        device,
        P,
        alpha,
        beta,
        net_optimizer,
        r_optimizer,
        inner_pbar,
        running_mean,
        decay=0.95,
        gamma=0.,  # 10.,
        loss='kl',
        regularization='kl',
        writer=None,
        update_tb=100,
        balanced=False,
        i=None):
    """Wrapper for running outer loop operations."""
    for bi in range(outer_batch_size_multiplier):
        batch, labels = P.sample_batch(batch_size)
        out = net(batch)
        net_cce = net_loss(out, labels)
        if adv_version == 'entropy' and loss != 'targeted':
            labels = torch.ones_like(out) / torch.tensor(num_classes).float()
        elif adv_version == 'entropy' and loss == 'targeted':
            labels = torch.eye(num_classes)[labels].float().to(device)
        else:
            raise NotImplementedError(adv_version)
        labels = labels.float()
        if loss == 'l2':
            Lo = torch.mean(torch.pow(out, 2))
        elif loss == 'l1':
            Lo = torch.mean(torch.abs(out))
        elif loss == 'huber':
            Lo = torch.nn.SmoothL1Loss(
                reduction='mean')(input=out, target=labels)
        elif loss == 'hinge':
            pos_loss = torch.mean(F.relu(1. - out))
            neg_loss = torch.mean(F.relu(1 - out * -1))
            Lo = pos_loss + neg_loss
        elif loss == 'kl' or loss is None:  # Default
            out = nn.LogSoftmax(dim=-1)(out)  # Use log probabilities
            Lo = torch.nn.KLDivLoss(
                reduction='batchmean')(input=out, target=labels)
        elif loss == 'margin':
            out = nn.ReLU()(out)  # Use log probabilities
            Lo = torch.mean(out)
        elif loss == 'targeted':
            # out = nn.ReLU()(out + 2) ** 2  # Use log probabilities
            Lo = torch.mean(F.softplus((out + 2)))
            # labels = roll(labels, shift=10)
            # Lo = torch.mean(torch.sum(labels * nn.Softmax(dim=-1)(out), 1))
            # Lo = Lo + torch.mean(F.softplus(torch.clamp(-out, -np.inf, 10)))
        elif loss == 'kl_prob':
            Lo = torch.mean(torch.sum(-labels * nn.LogSoftmax(dim=-1)(out), 1))
        else:
            raise NotImplementedError('Cannot find loss: {}'.format(loss))

        # Get generative losses
        if regularization == 'l2':
            generative_losses = get_z(
                dists=P.dists,
                P=P)
        elif regularization == 'kl':
            generative_losses = get_kl(
                dists=P.dists,
                P=P)
        else:
            raise NotImplementedError('Cannot find regularization: {}'.format(regularization))  # noqa
        if adv_version == 'flip':
            generative_losses *= -1

        # Combine losses
        Lo = alpha * Lo
        if balanced:
            beta = Lo / generative_losses
            beta = 0.1 * torch.floor(beta) + 1e-4
        generative_losses = generative_losses * beta
        r_loss = Lo + generative_losses

        # Add additional regularization for label embedding if using biggan
        record_emb = False
        if hasattr(P, 'embedding_grad') and P.embedding_grad:
            # Add an L2 penality on difference between original and new
            embedding = P.get_embed()[0][1] - P.embedding_original
            emb_loss = torch.norm(embedding, 1) * gamma
            record_emb = True
            r_loss = r_loss + emb_loss
        r_loss.backward()

    # Update params
    r_optimizer.step()
    pdict = dict(P.named_parameters())
    pclean = OrderedDict()
    if 0:  # P.dataset == 'psvrt':
        for n, p in list(pdict.items()):
            n = '_'.join([n1[:3] for n1 in n.split('_')])
            if p.shape == ():
                pclean[n] = p.detach().cpu().numpy().tolist()
            else:
                p2 = p.detach().cpu().numpy().tolist()[0]
                if isinstance(p2, list):
                    pclean[n] = p2[0]
                else:
                    pclean[n] = p2
        pstring = ' | '.join(['{}:{:.4f}'.format(ni, p) for ni, p in list(pclean.items())])  # noqa
    else:
        # loc = P.parameters()[0]
        # scale = P.parameters()[1]
        pstring = ''
    params = {}
    for k, v in pdict.items():
        if v.grad is not None:
            params[k] = v

    # Update status
    if running_mean == 0:
        running_mean = Lo
    else:
        running_mean = decay * running_mean + (1 - decay) * Lo
    if i is not None:
        desc = 'Class loss {:.4f}, total loss {:.4f}, running loss {:.4f}, {} {:.4f}, gen_kl {:.4f}\t'.format(  # noqa
            net_cce, r_loss, running_mean, loss, Lo, generative_losses)
    else:
        desc = 'Step:{}, class loss {:.4f}, total loss {:.4f}, running loss {:.4f}, {} {:.4f}, gen_kl {:.4f}\t'.format(  # noqa
            i, net_cce, r_loss, running_mean, loss, Lo, generative_losses)
    inner_pbar.set_description(desc + pstring)
    inner_pbar.update(1)
    if writer is not None and i % update_tb == 0:
        writer.add_scalar('Outer_loop/total', r_loss, i)
        writer.add_scalar('Outer_loop/classification_loss', net_cce, i)
        writer.add_scalar('Outer_loop/gen_KL', generative_losses, i)
        writer.add_scalar('Outer_loop/{}'.format(loss), Lo, i)
        for k, v in params.items():
            writer.add_histogram('Outer_loop/{}'.format(k), v.grad, i)
        if record_emb:
            writer.add_scalar('Outer_loop/embedding_diff', emb_loss, i)
        if P.dataset == 'psvrt':
            for k, v in params.items():
                if len(v.shape) == 0:
                    writer.add_scalar('Outer_loop/trace_{}'.format(k), v, i)
        images = get_examples(P=P, convert_to_numpy=False)
        writer.add_images(
            'Outer_loop/images',
            img_tensor=images.permute(0, 3, 1, 2),
            global_step=i)
    r_optimizer.zero_grad()
    net_optimizer.zero_grad()
    return Lo, generative_losses, r_loss, batch, running_mean, net_cce, params


def default_hps(name):
    """Default hyperparmas for different experiments."""
    hps = {}

    if name == 'biggan':
        hps['dataset'] = 'biggan'
        hps['batch_size'] = 22
        # hps['batch_size'] = 64
        hps['siamese'] = False
        hps['pretrained'] = True
        hps['inner_steps'] = 1.
        hps['inner_lr'] = 1e-5
        hps['outer_lr'] = 1e-2
        hps['wn'] = True
        hps['use_bn'] = False
        hps['load_model'] = None
        hps['beta'] = 0.001  # 0.5 KL modulator
    elif name == 'psvrt':
        hps['dataset'] = 'psvrt'
        hps['batch_size'] = 256
        hps['siamese'] = False
        hps['pretrained'] = False
        hps['inner_steps'] = 0.15  # 0.1
        hps['inner_lr'] = 1e-5
        hps['outer_lr'] = 5e-2
        # hps['load_model'] = 'psvrt.pth'
        hps['load_model'] = None
        hps['wn'] = False
        hps['use_bn'] = False
        hps['beta'] = 0.1  # KL modulator
        # hps['beta'] = 0.000001  # KL modulator
    else:
        raise NotImplementedError(name)

    # Dataset agnostic
    hps['model_name'] = 'resnet50'  # 'cbam'
    hps['inner_loop_criterion'] = True  # reach specified loss
    hps['inner_loop_nonfirst_criterion'] = False  # steps
    hps['outer_loop_criterion'] = False
    hps['outer_steps'] = 60000
    hps['inner_steps_first'] = hps['inner_steps']
    hps['inner_steps_nonfirst'] = 1  # < 1 is loop change criteria... i.e. train until 0.2 loss  # noqa
    hps['inner_lr_first'] = hps['inner_lr']
    hps['inner_lr_nonfirst'] = 1e-5  # < 1 is loop change criteria... i.e. train until 0.2 loss  # noqa
    hps['epochs'] = 1
    hps['alpha'] = 1.  # Main loss modulator
    hps['outer_batch_size_multiplier'] = 1  # Multiplier for outer steps
    hps['optimizer'] = 'Adam'
    hps['adv_version'] = 'entropy'  # 'entropy'  # 'flip'  # or 'entropy'
    hps['siamese_version'] = 'subtract'  # 'conv'
    hps['amsgrad'] = True
    hps['task'] = 'sd'  # sd (same diff) or sr (spatial relations)
    hps['results_dir'] = 'results'
    hps['gen_tb'] = True
    hps['loss'] = 'kl'  # kl, targeted
    hps['emb_lr'] = 1e-6  # 1e-3
    hps['save_i_params'] = 5000
    hps['regularization'] = 'kl'
    return hps

