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
from resources.pytorch_pretrained_biggan import one_hot_from_names
from matplotlib import pyplot as plt
torch.backends.cudnn.benchmark = True


def experiment(param_path, category, dataset_dir, steps, n, used_wn=True):
    """Run a dataset-adv experiment. Pull from DB or use defaults."""
    # Set default training parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(('Using device: {}'.format(device)))

    # Load dataset params and get name of dataset
    old_cat = np.copy(category)
    category = category.replace('_', ' ')
    assert os.path.exists(param_path), 'Could not find {}'.format(param_path)
    dataset = param_path.split(os.path.sep)[-1].split('_')[0]
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
        time.time()).strftime('%Y-%m-%d-%H_%M_%S')
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
        task='sd',
        wn=False,
        num_classes=num_classes)
    P = P.to(device)
    if used_wn:
        lambda_0_r = utils.normalize_weights(P=params, name='noise_vector', prop='center')
        lambda_0_scale_r = utils.normalize_weights(P=params, name='noise_vector', prop='scale')
        lambda_0_factor_r = utils.normalize_weights(P=params, name='noise_vector', prop='factor')
    else:
        raise NotImplementedError

    # Pull out original params
    class_vector = one_hot_from_names([category]).repeat(n, 0)
    lambda_0 = P.dists[0]['lambda_0'].cpu().numpy()
    lambda_0_scale = P.dists[0]['lambda_0_scale'].cpu().numpy()
    lambda_0_factor = P.dists[0]['lambda_0_factor'].cpu().numpy()
    interp = np.linspace(0, 1, steps)
    f = plt.figure()
    for b, itp in tqdm(enumerate(interp), desc='Interpolation steps', total=steps):
        # Do linear combinations of original and final
        i_lambda_0 = (1 - itp) * lambda_0 + itp * lambda_0_r
        i_lambda_scale_0 = (1 - itp) * lambda_0_scale + itp * lambda_0_scale_r
        i_lambda_factor_0 = (1 - itp) * lambda_0_factor + itp * lambda_0_factor_r
        it_params = {
            'noise_vector_center': i_lambda_0,
            'noise_vector_scale': i_lambda_scale_0,
            'noise_vector_factor': i_lambda_factor_0
        }
        with torch.no_grad():
            images, labels = P.sample_batch(n, force_params=it_params, class_vector=class_vector, force_mean=True)
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        norm_mean = P.norm_mean.cpu().numpy().reshape(1, 1, 3)
        norm_std = P.norm_std.cpu().numpy().reshape(1, 1, 3)
        images = images * norm_std + norm_mean
        row_ids = np.arange(b, steps * n, steps) + 1
        for idx, row in enumerate(row_ids):
            plt.subplot(n, steps, row)
            plt.axis('off')
            plt.imshow(images[idx])
    f.text(0.05, 0.5, 'Mean image -2 to +2 SDs', va='center', rotation='vertical')
    f.text(0.5, 0.04, 'Interpolation from start to optimized parameters', ha='center')
    plt.gcf().subplots_adjust(left=0.15)
    # plt.tight_layout()
    # plt.ylabel('Mean image -2 to +2 SDs')
    # plt.xlabel('Interpolation from start to optimized parameters')
    plt.savefig(os.path.join(ds_name, '{}.pdf'.format(old_cat)), dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--param_path',
        dest='param_path',
        type=str,
        default=None,
        help='Path to npz with dataset parameters.')
    parser.add_argument(
        '--cat',
        dest='category',
        type=str,
        default='border collie',
        help='Image category.')
    parser.add_argument(
        '--steps',
        dest='steps',
        type=int,
        default=7,
        help='Steps of interpolation.')
    parser.add_argument(
        '--n',
        dest='n',
        type=int,
        default=5,
        help='Rows of examples.')
    parser.add_argument(
        '--dataset_dir',
        dest='dataset_dir',
        type=str,
        default='interpolations',
        help='Output directory for interpolations.')
    args = parser.parse_args()
    start = time.time()
    experiment(**vars(args))
    end = time.time()
    print(('Experiment took {}'.format(end - start)))

