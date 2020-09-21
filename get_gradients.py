import os
import time
import argparse
import datetime
import torch
import utils
import numpy as np
from torch import nn
from torch import optim
from tqdm import tqdm
from collections import OrderedDict
from importlib import import_module
from db import db
# from torchviz import make_dot
torch.backends.cudnn.benchmark = True


def reversal(grad):
    """Reverse direction of gradients."""
    grad_clone = grad.clone()
    return grad_clone.neg()


def prep_params(params):
    """Prepare parameters for saving."""
    return {
        k: v.detach().cpu().numpy().tolist()
        for k, v in params.named_parameters()}


def outer_loop(
        batch_size,
        outer_batch_size_multiplier,
        adv_version,
        num_classes,
        net,
        net_loss,
        device,
        P,
        outer_steps,
        alpha,
        beta,
        net_optimizer,
        r_optimizer,
        inner_pbar,
        loss_version='kl',
        balanced=False,
        i=None):
    """Wrapper for running outer loop operations."""
    batch, labels = P.sample_batch(
        outer_batch_size_multiplier * batch_size)
    out = net(batch)
    if adv_version == 'entropy':
        labels = torch.ones_like(out) / torch.tensor(num_classes).float()
    labels = labels.float()

    if loss_version == 'l2':
        Lo = torch.mean(torch.pow(out, 2))
    elif loss_version == 'l1':
        Lo = torch.mean(torch.abs(out))
    elif loss_version == 'huber':
        Lo = torch.nn.SmoothL1Loss(reduction='mean')(input=out, target=labels)
    elif loss_version == 'kl':
        out = nn.LogSoftmax(dim=-1)(out)  # Use log probabilities
        Lo = torch.nn.KLDivLoss(
            reduction='batchmean')(input=out, target=labels)
    else:
        raise NotImplementedError(loss_version)

    # Get generative losses
    generative_losses = utils.get_kl(
        dists=P.dists,
        P=P)
    if adv_version == 'flip':
        generative_losses *= -1

    # Combine losses
    Lo = alpha * Lo
    generative_losses = generative_losses * beta
    r_loss = Lo + generative_losses

    # Gather dL/dLatents
    r_loss.backward(torch.tensor(1. / outer_steps).to(P.device))
    grads = {k: v for k, v in P.named_parameters()}
    # net_optimizer.zero_grad()

    # Update status
    inner_pbar.update(1)
    return Lo, generative_losses, r_loss, grads


def inner_loop(
        net,
        net_loss,
        net_optimizer,
        P,
        device,
        inner_pbar,
        batch_size):
    """Run model optim."""
    with torch.no_grad():
        batch, labels = P.sample_batch(batch_size)
    out = net(batch)
    L = net_loss(out, labels)
    L.backward()
    net_optimizer.step()
    net_optimizer.zero_grad()
    desc = 'Loss is {:.4f}'.format(L)
    inner_pbar.set_description(desc)
    inner_pbar.update(1)
    return L


def experiment(**kwargs):
    """Run a dataset-adv experiment. Pull from DB or use defaults."""
    # Set default training parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(('Using device: {}'.format(device)))
    hps = utils.default_hps('biggan')
    hps['epochs'] = 1

    # Update params with kwargs
    pull_from_db = kwargs['pull_from_db']
    if pull_from_db:
        exp = db.get_experiment_trial(False)
        if exp is None:
            raise RuntimeError('All experiments are complete.')
    else:
        exp = kwargs
    for k, v in exp.items():
        if k in hps:
            hps[k] = v
            print(('Setting {} to {}'.format(k, v)))

    # Create results directory
    utils.make_dir(hps['results_dir'])

    # Other params we won't write to DB
    save_examples = False
    im_dir = 'screenshots'
    trainable = True
    reset_inner_optimizer = True  # Reset adam params after every epoch
    if hps['dataset'] == 'biggan':
        num_classes = 1000
        model_output = 1000
    elif hps['dataset'] == 'psvrt':
        num_classes = 2
        model_output = 2
    else:
        raise NotImplementedError(hps['dataset'])
    net_loss = nn.CrossEntropyLoss(reduction='mean')

    # Create results directory
    utils.make_dir(hps['results_dir'])

    # Add hyperparams and model info to DB
    dt = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    hps['dt'] = dt
    run_name = '{}_{}_{}'.format(hps['dataset'], hps['model_name'], dt)

    # Initialize net
    net, img_size = utils.initialize_model(
        dataset=hps['dataset'],
        model_name=hps['model_name'],
        num_classes=model_output,
        siamese=hps['siamese'],
        siamese_version=hps['siamese_version'],
        trainable=trainable,
        pretrained=hps['pretrained'])
    if hps['dataset'] == 'biggan':
        img_size = 224
    elif hps['dataset'] == 'psvrt':
        img_size = 80  # 160

    ds = import_module('data_generators.{}'.format(hps['dataset']))
    P = ds.Generator(
        dataset=hps['dataset'],
        img_size=img_size,
        device=device,
        siamese=hps['siamese'],
        task=hps['task'],
        wn=hps['wn'],
        num_classes=num_classes)
    if hps['adv_version'] == 'flip':
        [p.register_hook(reversal) for p in P.parameters()]

    net = net.to(device)
    P = P.to(device)
    net_optimizer = utils.get_optimizer(
        net=net,
        optimizer=hps['optimizer'],
        lr=hps['inner_lr'],
        amsgrad=hps['amsgrad'],
        trainable=trainable)
    if hps['dataset'] == 'biggan':
        outer_params = P.named_parameters()
        outer_params = [v for k, v in outer_params if 'model' not in k]
    else:
        outer_params = P.parameters()
    r_optimizer = getattr(optim, hps['optimizer'])(
        outer_params, lr=hps['outer_lr'], amsgrad=hps['amsgrad'])

    # Optimize r
    inner_losses, outer_losses = [], []
    inner_loop_steps, outer_loop_steps = [], []
    epochs = int(hps['epochs'])
    inner_loop_criterion = hps['inner_loop_criterion']
    outer_loop_criterion = hps['outer_loop_criterion']
    if inner_loop_criterion:
        inner_steps = hps['inner_steps']
    else:
        inner_steps = int(hps['inner_steps'])
    if outer_loop_criterion:
        outer_steps = hps['outer_steps']
    else:
        outer_steps = int(hps['outer_steps'])
    for epoch in tqdm(
            list(range(epochs)),
            total=epochs,
            desc='Epoch'):
        # Inner loop starts here
        net.train()
        net._initialize()  # Reset thetas
        P.set_not_trainable()
        if reset_inner_optimizer:
            if epoch == 0:
                reset_adam_state = net_optimizer.state
            net_optimizer.state = reset_adam_state  # Reset adam parameters
        with tqdm(total=inner_steps) as inner_pbar:
            if inner_loop_criterion:
                L = np.inf
                i = 0
                while L > inner_steps:
                    L = inner_loop(
                        net=net,
                        net_loss=net_loss,
                        net_optimizer=net_optimizer,
                        P=P,
                        device=device,
                        inner_pbar=inner_pbar,
                        batch_size=hps['batch_size'])
                    i += 1
            else:
                for i in range(inner_steps):
                    L = inner_loop(
                        net=net,
                        net_loss=net_loss,
                        net_optimizer=net_optimizer,
                        P=P,
                        device=device,
                        inner_pbar=inner_pbar,
                        batch_size=hps['batch_size'])
            inner_loop_steps += [i]

        # TODO: Compute hessian over init_training_steps here
        # TODO: Pass adams from inner_optimizer to r_optimizer
        inner_losses += [L.cpu().data.numpy()]

        # Outer loop starts here
        net.eval()  # Enable test-time batch norms
        P.set_trainable()
        if save_examples:
            utils.plot_examples(
                path=os.path.join(
                    im_dir, '{}_outer_init_{}'.format(dt, epoch)),
                n=16,
                P=P)
        with tqdm(total=outer_steps) as inner_pbar:
            if outer_loop_criterion:
                L = np.inf
                i = 0
                while L > outer_steps:
                    Lo, generative_losses, r_loss, grads = outer_loop(
                        batch_size=hps['batch_size'],
                        outer_batch_size_multiplier=hps['outer_batch_size_multiplier'],  # noqa
                        adv_version=hps['adv_version'],
                        num_classes=num_classes,
                        net_optimizer=net_optimizer,
                        r_optimizer=r_optimizer,
                        net=net,
                        net_loss=net_loss,
                        device=device,
                        loss=hps['loss'],
                        P=P,
                        outer_steps=outer_steps,
                        alpha=hps['alpha'],
                        beta=hps['beta'],
                        inner_pbar=inner_pbar)
                    i += 1
            else:
                for i in range(outer_steps):
                    Lo, generative_losses, r_loss, grads = outer_loop(
                        batch_size=hps['batch_size'],
                        outer_batch_size_multiplier=hps['outer_batch_size_multiplier'],  # noqa
                        adv_version=hps['adv_version'],
                        num_classes=num_classes,
                        net_optimizer=net_optimizer,
                        r_optimizer=r_optimizer,
                        net=net,
                        net_loss=net_loss,
                        device=device,
                        loss=hps['loss'],
                        P=P,
                        outer_steps=outer_steps,
                        alpha=hps['alpha'],
                        beta=hps['beta'],
                        i=i,
                        inner_pbar=inner_pbar)
            outer_losses += [Lo]

        path = os.path.join(hps['results_dir'], '{}_gradients'.format(run_name))
        if pull_from_db:
            # Update DB
            results_dict = {
                '_id': exp['_id'],
                'file_path': path
            }
            db.update_grad_experiment([results_dict])

        # Save epoch results
        if save_examples:
            utils.plot_examples(
                path=os.path.join(
                    im_dir, '{}_outer_optim_{}'.format(run_name, epoch)),
                n=16,
                P=P)
        for k, v in grads.items():
            if v is not None:
                try:
                    v = v.detach().cpu().numpy()
                except Exception as e:
                    print('Failed to detach {}'.format(k))
                    v = v.cpu().numpy()
            grads[k] = v
        grads.update(hps)
        np.savez(
            path,
            **grads)
    print('Finished the experiment!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pull_from_db',
        dest='pull_from_db',
        action='store_true',
        help='Pull an experiment from the DB.')
    # parser.add_argument(
    #     '--siamese',
    #     dest='siamese',
    #     action='store_true',
    #     help='Use siamese net.')
    # parser.add_argument(
    #     '--task',
    #     dest='task',
    #     type=str,
    #     default='sd',
    #     help='sd or sr')
    # parser.add_argument(
    #     '--results_dir',
    #     dest='results_dir',
    #     type=str,
    #     default='results',
    #     help='Output directory for results.')
    args = parser.parse_args()
    start = time.time()
    experiment(**vars(args))
    end = time.time()
    print(('Experiment took {}'.format(end - start)))

