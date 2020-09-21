import os
import time
import argparse
import datetime
import torch
import utils
import json
import numpy as np
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
from db import db
# from torchviz import make_dot
torch.backends.cudnn.benchmark = True


def experiment(**kwargs):
    """Run a dataset-adv experiment. Pull from DB or use defaults."""
    # Set default training parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(('Using device: {}'.format(device)))
    hps = utils.default_hps('biggan')
    hps = utils.default_hps('psvrt')
    hps['epochs'] = 1

    # Update params with kwargs
    pull_from_db = kwargs['pull_from_db']
    if pull_from_db:
        exp = db.get_experiment_trial(True)
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
    save_every = 1000
    save_examples = False
    im_dir = 'screenshots'
    trainable = True
    reset_inner_optimizer = False  # Reset adam params after every epoch
    reset_theta = False
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
        time.time()).strftime('%Y-%m-%d-%H_%M_%S')
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
        # net.track_running_stats = False
    elif hps['dataset'] == 'psvrt':
        img_size = 160

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
        [p.register_hook(utils.reversal) for p in P.parameters()]

    net = net.to(device)
    P = P.to(device)
    net_optimizer = utils.get_optimizer(
        net=net,
        optimizer=hps['optimizer'],
        lr=hps['inner_lr'],
        amsgrad=hps['amsgrad'],
        trainable=trainable)
    if hps['dataset'] == 'biggan':
        outer_params = [v for k, v in P.named_parameters() if 'model' not in k]
        if P.embedding_grad:
            outer_params = [
                {'params': outer_params},
                {'params': P.get_embed()[0][1], 'lr': hps['emb_lr']}
            ]
            # outer_params += [P.embedding]
    else:
        outer_params = P.parameters()
    r_optimizer = getattr(optim, hps['optimizer'])(
        outer_params,
        lr=hps['outer_lr'],
        amsgrad=hps['amsgrad'])

    # Add tensorboard if requested
    if hps['gen_tb']:
        writer = SummaryWriter(log_dir=os.path.join('runs', run_name))
        print(
            'Saving tensorboard to: {}'.format(os.path.join('runs', run_name)))
    else:
        writer = None

    # Optimize r
    inner_losses, outer_losses = [], []
    inner_loop_steps, outer_loop_steps = [], []
    all_params = []
    epochs = int(hps['epochs'])
    inner_loop_criterion = hps['inner_loop_criterion']
    outer_loop_criterion = hps['outer_loop_criterion']
    if hps['inner_loop_criterion']:
        inner_steps = hps['inner_steps']
    else:
        inner_steps = int(hps['inner_steps'])
    if hps['outer_loop_criterion']:
        outer_steps = hps['outer_steps']
    else:
        outer_steps = int(hps['outer_steps'])
    for epoch in tqdm(
            list(range(epochs)),
            total=epochs,
            desc='Epoch'):
        # Inner loop starts here
        net.train()
        P.set_not_trainable()
        if reset_theta:
            net._initialize()
        if reset_inner_optimizer:
            if epoch == 0:
                reset_adam_state = net_optimizer.state
            net_optimizer.state = reset_adam_state  # Reset adam parameters
        with tqdm() as inner_pbar:
            if inner_loop_criterion:
                L = np.inf
                i = 0
                while L > inner_steps:
                    L = utils.inner_loop(
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
                    L = utils.inner_loop(
                        net=net,
                        net_loss=net_loss,
                        net_optimizer=net_optimizer,
                        P=P,
                        device=device,
                        inner_pbar=inner_pbar,
                        batch_size=hps['batch_size'])
            inner_loop_steps += [i]

        # TODO: Pass adams from inner_optimizer to r_optimizer
        inner_losses += [L.item()]  # cpu().data.numpy()]

        # Outer loop starts here
        if hps['use_bn']:
            net.eval()  # Careful!!
        else:
            net.train()
        P.set_trainable()
        if save_examples:
            utils.plot_examples(
                path=os.path.join(
                    im_dir, '{}_outer_init_{}'.format(run_name, epoch)),
                n_subplots=16,
                n_batches=10,
                P=P)
        try:
            with tqdm() as inner_pbar:
                rmu = 0.
                if outer_loop_criterion:
                    L = np.inf
                    i = 0
                    while L > outer_steps:
                        Lo, generative_losses, r_loss, batch, rmu, net_ce, params = utils.outer_loop(  # noqa
                            batch_size=hps['batch_size'],
                            outer_batch_size_multiplier=hps['outer_batch_size_multiplier'],  # noqa
                            adv_version=hps['adv_version'],
                            num_classes=num_classes,
                            net_optimizer=net_optimizer,
                            r_optimizer=r_optimizer,
                            net=net,
                            net_loss=net_loss,
                            running_mean=rmu,
                            device=device,
                            loss=hps['loss'],
                            P=P,
                            alpha=hps['alpha'],
                            beta=hps['beta'],
                            writer=writer,
                            i=i,
                            inner_pbar=inner_pbar)
                        if (
                                hps['save_i_params'] and
                                i % hps['save_i_params'] == 0):
                            all_params += [utils.prep_params(params)]
                        i += 1
                else:
                    for i in range(outer_steps):
                        Lo, generative_losses, r_loss, batch, rmu, net_ce, params = utils.outer_loop(  # noqa
                            batch_size=hps['batch_size'],
                            outer_batch_size_multiplier=hps['outer_batch_size_multiplier'],  # noqa
                            adv_version=hps['adv_version'],
                            num_classes=num_classes,
                            net_optimizer=net_optimizer,
                            r_optimizer=r_optimizer,
                            net=net,
                            net_loss=net_loss,
                            running_mean=rmu,
                            device=device,
                            loss=hps['loss'],
                            P=P,
                            alpha=hps['alpha'],
                            beta=hps['beta'],
                            i=i,
                            writer=writer,
                            inner_pbar=inner_pbar)
                        if (
                                hps['save_i_params'] and
                                i % hps['save_i_params'] == 0):
                            all_params += [utils.prep_params(params)]
                outer_losses += [Lo.item()]  # cpu().data.numpy()]
                outer_loop_steps += [i]
        except Exception as e:
            print(
                'Outer optimization failed. {}\n'
                'Saving results and exiting.'.format(e))
            if pull_from_db:
                # Update DB with results
                results_dict = {
                    'experiment_id': exp['_id'],
                    'inner_loss': inner_losses[epoch].tolist(),
                    'outer_loss': outer_losses[epoch].tolist(),
                    'inner_loop_steps': inner_loop_steps[epoch],
                    'outer_loop_steps': outer_loop_steps[epoch],
                    'net_loss': net_ce.item(),  # cpu().data.numpy(),
                    'params': json.dumps(utils.prep_params(P)),
                }
                db.add_results([results_dict])
            break

        # Save epoch results
        if save_examples:
            utils.plot_examples(
                path=os.path.join(
                    im_dir,
                    '{}_outer_optim_{}'.format(run_name, epoch)),
                n_subplots=16,
                n_batches=10,
                P=P)
        # pds += [utils.prep_params(P)]
        if epoch % save_every == 0:
            np.save(
                os.path.join(
                    hps['results_dir'],
                    '{}_inner_losses'.format(run_name)),
                inner_losses)
            np.save(
                os.path.join(
                    hps['results_dir'],
                    '{}_outer_losses'.format(run_name)),
                outer_losses)
            save_params = utils.prep_params(P)
            if P.embedding_grad:
                save_params['embedding'] = P.get_embed()[0][1].detach().cpu().numpy()  # noqa
                save_params['embedding_original'] = P.embedding_original.detach().cpu().numpy()  # noqa
            np.save(
                os.path.join(
                    hps['results_dir'],
                    '{}_all_params'.format(run_name)),
                all_params)
            np.savez(
                os.path.join(
                    hps['results_dir'],
                    '{}_final_params'.format(run_name)),
                **save_params)

        if len(inner_loop_steps):
            np.save(
                os.path.join(
                    hps['results_dir'],
                    '{}_inner_steps'.format(run_name)),
                inner_loop_steps)
            np.save(
                os.path.join(
                    hps['results_dir'],
                    '{}_outer_steps'.format(run_name)),
                outer_loop_steps)
        if pull_from_db:
            # Update DB with results
            results_dict = {
                'experiment_id': exp['_id'],
                'inner_loss': inner_losses[epoch].tolist(),
                'outer_loss': outer_losses[epoch].tolist(),
                'inner_loop_steps': inner_loop_steps[epoch],
                'outer_loop_steps': outer_loop_steps[epoch],
                'net_loss': net_ce.item(),  # cpu().data.numpy(),
                'params': json.dumps(utils.prep_params(P)),
            }
            db.add_results([results_dict])
    print('Finished {}!'.format(run_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pull_from_db',
        dest='pull_from_db',
        action='store_true',
        help='Pull an experiment from the DB.')
    args = parser.parse_args()
    start = time.time()
    experiment(**vars(args))
    end = time.time()
    print(('Experiment took {}'.format(end - start)))

