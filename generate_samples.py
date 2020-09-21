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
from importlib import import_module
from resources.pytorch_pretrained_biggan.utils import IMAGENET
torch.backends.cudnn.benchmark = True


REV_IMAGENET = {v: k for k, v in IMAGENET.items()}


def experiment(param_path, n, batch, dataset_dir, used_wn=True):
    """Run a dataset-adv experiment. Pull from DB or use defaults."""
    # Set default training parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(('Using device: {}'.format(device)))

    # Load dataset params and get name of dataset
    assert os.path.exists(param_path), 'Could not find {}'.format(param_path)
    dataset = param_path.replace('lagrange_', '').split(os.path.sep)[-1].split('_')[0]
    params = np.load(param_path)
    if used_wn:
        # Dataset optimized using weight norm
        if dataset == 'biggan':
            params = {
                'noise_vector_center_g': params.f.noise_vector_center_g,
                'noise_vector_center_v': params.f.noise_vector_center_v,
                'noise_vector_scale_g': params.f.noise_vector_scale_g,
                'noise_vector_scale_v': params.f.noise_vector_scale_v,
                'noise_vector_factor_g': params.f.noise_vector_factor_g,
                'noise_vector_factor_v': params.f.noise_vector_factor_v,
            }
        elif dataset == 'psvrt':
            raise NotImplementedError
    else:
        if dataset == 'biggan':
            params = {
                'noise_vector_center': params.f.noise_vector_center,
                'noise_vector_scale': params.f.noise_vector_scale,
                'noise_vector_factor': params.f.noise_vector_factor
            }
        elif dataset == 'psvrt':
            raise NotImplementedError
    if dataset == 'biggan':
        num_classes = 1000
    elif dataset == 'psvrt':
        num_classes = 2
    else:
        raise NotImplementedError(dataset)

    # Add hyperparams and model info to DB
    dt = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    ds_name = os.path.join(dataset_dir, '{}_{}'.format(dataset, dt))

    # Create results directory
    utils.make_dir(dataset_dir)
    utils.make_dir(ds_name)

    # Initialize dataset object
    if dataset == 'biggan':
        img_size = 256
    elif dataset == 'psvrt':
        img_size = 80  # 160
    ds = import_module('data_generators.{}'.format(dataset))
    P = ds.Generator(
        dataset=dataset,
        img_size=img_size,
        device=device,
        siamese=False,
        crop=None,
        task='sd',
        wn=True,
        num_classes=num_classes)
    P = P.to(device)
    ln = n // batch
    assert ln > 0, 'Producing 0 images per batch.'
    for b in tqdm(range(batch), desc='Saving image batches', total=batch):
        with torch.no_grad():
            images, labels = P.sample_batch(ln, force_params=params)
        if P.siamese:
            images = images.sum(-1)
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        norm_mean = P.norm_mean.cpu().numpy().reshape(1, 1, 3)
        norm_std = P.norm_std.cpu().numpy().reshape(1, 1, 3)
        images = images * norm_std + norm_mean
        labels = labels.detach().cpu().numpy()
        if ln == 1:
            synset = REV_IMAGENET[labels.tolist()]
        else:
            synset = []
            for label in labels:
                synset += [REV_IMAGENET[label]]
        np.savez(
            os.path.join(ds_name, str(b)),
            images=images,
            synset=np.array(synset),
            labels=labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--param_path',
        dest='param_path',
        type=str,
        default=None,
        help='Path to npz with dataset parameters.')
    parser.add_argument(
        '--n',
        dest='n',
        type=int,
        default=100000,
        help='Number of images to produce.')
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help='Split total n into this many npzs.')
    parser.add_argument(
        '--dataset_dir',
        dest='dataset_dir',
        type=str,
        default='datasets',
        help='Output directory for datasets.')
    args = parser.parse_args()
    start = time.time()
    experiment(**vars(args))
    end = time.time()
    print(('Experiment took {}'.format(end - start)))

