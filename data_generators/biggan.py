import torch
import numpy as np
from torch import nn
from resources.pytorch_pretrained_biggan import BigGAN
from resources.pytorch_pretrained_biggan import one_hot_from_names
from resources.pytorch_pretrained_biggan import truncated_noise_sample
from resources.pytorch_pretrained_biggan import save_as_images
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal  # noqa
# from torch.distributions import constraints


# BigGan parameters
NV_DIM = 128
DIST_BOUNDS = [-2, 2]
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
TRUNCATION = 0.6  # 0.6 works for 128
# BIGGAN_TYPE = 'biggan-deep-128'
CROP = 'none'
BIGGAN_TYPE = 'biggan-deep-256'
# CROP = 'center'

FAMILY = 'normal'
# FAMILY = 'mv_normal'
# FAMILY = 'low_mv_normal'
if FAMILY == 'normal':
    OBJECT_NUMBER_LOC = torch.tensor(0.).repeat(NV_DIM).float()
    OBJECT_NUMBER_SCALE = torch.tensor(TRUNCATION).repeat(NV_DIM).float()
    DISTS = [{
        'name': 'noise_vector',
        'family': FAMILY,
        'lambda_0': OBJECT_NUMBER_LOC,
        'lambda_0_scale': OBJECT_NUMBER_SCALE,
        'return_sampler': True,
        'trainable': True,
    }]
elif FAMILY == 'low_mv_normal':
    F_DIM = 16  # 64
    OBJECT_NUMBER_LOC = torch.tensor(0.).repeat(NV_DIM).float()
    OBJECT_NUMBER_SCALE = torch.tensor(TRUNCATION).repeat(NV_DIM).float()
    OBJECT_NUMBER_FACTOR = torch.ones((NV_DIM, F_DIM)).float()
    OBJECT_NUMBER_FACTOR[-1] = 0.
    DISTS = [{
        'name': 'noise_vector',
        'family': FAMILY,
        'lambda_0': OBJECT_NUMBER_LOC,
        'lambda_0_scale': OBJECT_NUMBER_SCALE,
        'lambda_0_factor': OBJECT_NUMBER_FACTOR,
        'return_sampler': True,
        'trainable': True,
    }]
elif FAMILY == 'mv_normal':
    OBJECT_NUMBER_LOC = torch.tensor(0.).repeat(NV_DIM).float()
    OBJECT_NUMBER_SCALE = torch.tensor(
        np.eye(NV_DIM, NV_DIM) * NV_DIM).float()
    MIN_OBJECT_NUMBER_SCALE = 2.5
    DISTS = [{
        'name': 'noise_vector',
        'family': FAMILY,
        'lambda_0': OBJECT_NUMBER_LOC,
        'lambda_0_scale': OBJECT_NUMBER_SCALE,
        'return_sampler': True,
        'trainable': True,
    }]


class Generator(nn.Module):
    def __init__(
            self,
            dataset,
            img_size,
            device,
            num_classes,
            siamese=False,  # Not used
            task='',  # Not used
            nv_loc=OBJECT_NUMBER_LOC,
            nv_scale=OBJECT_NUMBER_SCALE,
            truncation=TRUNCATION,
            biggan_type=BIGGAN_TYPE,
            embedding_grad=False,
            wn=False,
            crop=CROP,
            # crop='center'
            batch_grad=False,
            learn_truncation=False,
            norm_mean=NORM_MEAN,
            norm_std=NORM_STD,
            dist_bounds=DIST_BOUNDS):
        """BigGAN image generator. Most of the above args are unused."""
        super(Generator, self).__init__()
        self.dataset = dataset
        self.img_size = img_size
        self.device = device
        self.truncation = truncation
        self.num_classes = num_classes
        self.wn = wn
        self.crop = crop
        self.siamese = siamese
        self.norm_mean = norm_mean.to(self.device)
        self.norm_std = norm_std.to(self.device)
        self.batch_grad = batch_grad
        self.embedding_grad = embedding_grad
        self.learn_truncation = learn_truncation
        self.model = BigGAN.from_pretrained(biggan_type)
        self.model.to(self.device)
        # Do not train the BigGAN
        for param in self.model.parameters():
            param.requires_grad = False

        # Prepare embedding attr if requested
        if self.embedding_grad:
            self.prepare_embed()

        # Use object_number as init for vector
        self.nv_scale = nv_scale
        self.dist_bounds = torch.tensor(dist_bounds).to(self.device)

        # Specify the distributions
        self.dists = DISTS
        self.init_challenge()

        # Init trunctation if requested
        if self.learn_truncation:
            self.truncation = nn.Parameter(
                torch.tensor(self.truncation),
                requires_grad=self.batch_grad).to(self.device)

    def check_dists(self):
        """Simple parameter checking for dists."""
        for di in self.dists:
            keys = list(di.keys())
            assert 'name' in keys, 'Need a name for your dist.'
            assert 'trainable' in keys, 'Need a trainable for your dist.'
            assert 'family' in keys, 'Need a family for your dist.'
            assert 'lambda_0' in keys, 'Need a lambda_0 for your dist.'
            assert 'return_sampler' in keys, 'Need a return_sampler for your dist.'  # noqa

    def normalize_weights(self, name, prop):
        """Apply weight normalization."""
        g_attr_name = '{}_{}'.format(name, '{}_g'.format(prop))
        v_attr_name = '{}_{}'.format(name, '{}_v'.format(prop))
        g = getattr(self, g_attr_name)
        v = getattr(self, v_attr_name)
        return v * (g / torch.norm(v)).expand_as(v)

    def init_wns(self, w):
        """Initialize weight normed weights."""
        g = torch.norm(w)
        v = w / g.expand_as(w)
        return g, v

    def _test(self, n=3, truncation=0.4, save_ims=True):
        """Produce n images."""
        print('Generating test images.')
        cats = ['border collie', 'mushroom']
        bs = len(cats)
        class_vector = one_hot_from_names(
            ['border collie', 'mushroom'], batch_size=bs)
        noise_vector = truncated_noise_sample(
            truncation=truncation, batch_size=bs)

        # All in tensors
        noise_vector = torch.from_numpy(noise_vector)
        class_vector = torch.from_numpy(class_vector)

        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to(self.device)
        class_vector = class_vector.to(self.device)

        # Generate an image
        with torch.no_grad():
            output = self.model(noise_vector, class_vector, truncation)

        # If you have a GPU put back on CPU
        output = output.to('cpu')
        if save_ims:
            save_as_images(output)
        return output

    def init_challenge(self, mu=1e-4, sigma=1e-4, pi=1e-4, eps=1e-4):
        """Initialize the perturbation vector r.
        We want to learn r, which perturbs the fixed random parameters
        \lambda_0. To do this, we will initialize r as \lambda_0, and
        use a generative loss to ensure r is minimally different than
        \lambda_0.

        Initialization is lambda_0 + alpha * distribution_noise.
        """
        for idx, row in enumerate(self.dists):
            trainable = row['trainable']
            # We have lambda_0 (for center/maybe scale)
            if not torch.is_tensor(row['lambda_0']):
                # If this is not a torch tensor, convert it
                row['lambda_0'] = torch.tensor(row['lambda_0'])

            # Send lambda_0 center to device
            row['lambda_0'] = row['lambda_0'].to(self.device)
            if (
                    row['family'] == 'gaussian' or
                    row['family'] == 'normal' or
                    row['family'] == 'mv_normal' or
                    row['family'] == 'abs_normal'):
                if not torch.is_tensor(row['lambda_0_scale']):
                    # If this is not a torch tensor, convert it
                    row['lambda_0_scale'] = torch.is_tensor(row['lambda_0_scale'])  # noqa
                # Send lambda_0 scale to device
                self.dists[idx]['lambda_0_scale'] = row['lambda_0_scale'].to(self.device)  # noqa

                # Initilize challenge center/scale
                lambda_r = row['lambda_0'] + torch.randn_like(row['lambda_0']) * mu  # noqa
                lambda_r_scale = row['lambda_0_scale'] + torch.randn_like(row['lambda_0_scale']) * sigma  # noqa

                # Also add the r parameters to a list
                if self.wn:
                    lambda_r_g, lambda_r_v = self.init_wns(lambda_r)
                    lambda_r_scale_g, lambda_r_scale_v = self.init_wns(lambda_r_scale)  # noqa

                    # Also add the r parameters to a list
                    attr_name = '{}_{}'.format(row['name'], 'center_g')
                    setattr(self, attr_name, nn.Parameter(lambda_r_g, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'center_v')
                    setattr(self, attr_name, nn.Parameter(lambda_r_v, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'scale_g')
                    setattr(self, attr_name, nn.Parameter(lambda_r_scale_g, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'scale_v')
                    setattr(self, attr_name, nn.Parameter(lambda_r_scale_v, requires_grad=trainable))  # noqa
                else:
                    # Also add the r parameters to a list
                    attr_name = '{}_{}'.format(row['name'], 'center')
                    setattr(self, attr_name, nn.Parameter(lambda_r, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'scale')
                    setattr(self, attr_name, nn.Parameter(lambda_r_scale, requires_grad=trainable))  # noqa
            elif row['family'] == 'low_mv_normal':
                # Send lambda_0 scale to device
                self.dists[idx]['lambda_0_scale'] = row['lambda_0_scale'].to(self.device)  # noqa
                self.dists[idx]['lambda_0_factor'] = row['lambda_0_factor'].to(self.device)  # noqa

                # Initilize challenge center/scale
                lambda_r = row['lambda_0'] + torch.randn_like(row['lambda_0']) * mu  # noqa
                lambda_r_scale = row['lambda_0_scale'] + torch.randn_like(row['lambda_0_scale']) * sigma  # noqa
                lambda_r_factor = row['lambda_0_factor'] + torch.randn_like(row['lambda_0_factor']) * sigma  # noqa

                # Also add the r parameters to a list
                if self.wn:
                    lambda_r_g, lambda_r_v = self.init_wns(lambda_r)
                    lambda_r_scale_g, lambda_r_scale_v = self.init_wns(lambda_r_scale)  # noqa
                    lambda_r_factor_g, lambda_r_factor_v = self.init_wns(lambda_r_factor)  # noqa

                    # Also add the r parameters to a list
                    attr_name = '{}_{}'.format(row['name'], 'center_g')
                    setattr(self, attr_name, nn.Parameter(lambda_r_g, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'center_v')
                    setattr(self, attr_name, nn.Parameter(lambda_r_v, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'scale_g')
                    setattr(self, attr_name, nn.Parameter(lambda_r_scale_g, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'scale_v')
                    setattr(self, attr_name, nn.Parameter(lambda_r_scale_v, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'factor_g')
                    setattr(self, attr_name, nn.Parameter(lambda_r_factor_g, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'factor_v')
                    setattr(self, attr_name, nn.Parameter(lambda_r_factor_v, requires_grad=trainable))  # noqa
                else:
                    # Also add the r parameters to a list
                    attr_name = '{}_{}'.format(row['name'], 'center')
                    setattr(self, attr_name, nn.Parameter(lambda_r, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'scale')
                    setattr(self, attr_name, nn.Parameter(lambda_r_scale, requires_grad=trainable))  # noqa
                    attr_name = '{}_{}'.format(row['name'], 'factor')
                    setattr(self, attr_name, nn.Parameter(lambda_r_factor, requires_grad=trainable))  # noqa
            else:
                raise NotImplementedError

    def sample_lambda0_r(
            self,
            d,
            batch_size,
            offset=0,
            gau=None):
        """Sample dataset parameters perturbed by r."""
        name = d['name']
        family = d['family']
        if self.wn:
            lambda_r = self.normalize_weights(name=name, prop='center')
        else:
            attr_name = '{}_{}'.format(name, 'center')
            lambda_r = getattr(self, attr_name)
        parameters = []
        if family == 'normal':
            attr_name = '{}_{}'.format(name, 'scale')
            if self.wn:
                lambda_r_scale = self.normalize_weights(name=name, prop='scale')  # noqa
            else:
                lambda_r_scale = getattr(self, attr_name)
            nor = Normal(loc=lambda_r, scale=lambda_r_scale)

            if d['return_sampler']:
                return nor
            else:
                for idx in range(batch_size):
                    parameters.append(nor.rsample())
        elif family == 'mv_normal':
            attr_name = '{}_{}'.format(name, 'scale')
            if self.wn:
                lambda_r_scale = self.normalize_weights(name=name, prop='scale')  # noqa
            else:
                lambda_r_scale = getattr(self, attr_name)
            nor = MultivariateNormal(loc=lambda_r, covariance_matrix=lambda_r_scale)  # noqa
            if d['return_sampler']:
                return nor
            else:
                for idx in range(batch_size):
                    parameters.append(nor.rsample())
        elif family == 'low_mv_normal':
            if self.wn:
                lambda_r_scale = self.normalize_weights(name=name, prop='scale')  # noqa
                lambda_r_factor = self.normalize_weights(name=name, prop='factor')  # noqa
            else:
                attr_name = '{}_{}'.format(name, 'scale')
                lambda_r_scale = getattr(self, attr_name)
                attr_name = '{}_{}'.format(name, 'factor')
                lambda_r_factor = getattr(self, attr_name)
            nor = LowRankMultivariateNormal(loc=lambda_r, cov_diag=lambda_r_scale, cov_factor=lambda_r_factor)  # noqa
            if d['return_sampler']:
                return nor
            else:
                for idx in range(batch_size):
                    parameters.append(nor.rsample())
        else:
            raise NotImplementedError(
                '{} not implemented in sampling.'.format(family))
        return parameters

    def force_parameters(self, force_params):
        """Set parameters to forced values."""
        if self.embedding_grad:
            import ipdb;ipdb.set_trace()
            # Set the embedding weights to the learned ones
        for idx, row in enumerate(self.dists):
            if self.wn:
                attr_name = '{}_{}'.format(row['name'], 'center_g')
                lambda_r_g = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_g)  # noqa
                attr_name = '{}_{}'.format(row['name'], 'center_v')
                lambda_r_v = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_v)  # noqa
                attr_name = '{}_{}'.format(row['name'], 'scale_g')
                lambda_r_scale_g = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_scale_g)  # noqa
                attr_name = '{}_{}'.format(row['name'], 'scale_v')
                lambda_r_scale_v = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_scale_v)  # noqa
            else:
                attr_name = '{}_{}'.format(row['name'], 'center')
                lambda_r = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r)  # noqa
                attr_name = '{}_{}'.format(row['name'], 'scale')
                lambda_r_scale = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_scale)  # noqa
            if row['family'] == 'low_mv_normal' and self.wn:
                attr_name = '{}_{}'.format(row['name'], 'factor_g')
                lambda_r_factor_g = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_factor_g)  # noqa
                attr_name = '{}_{}'.format(row['name'], 'factor_v')
                lambda_r_factor_v = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_factor_v)  # noqa
            elif row['family'] == 'low_mv_normal' and not self.wn:
                attr_name = '{}_{}'.format(row['name'], 'factor')
                lambda_r_factor = nn.Parameter(
                    torch.tensor(
                        force_params[attr_name]).float().to(self.device))
                setattr(self, attr_name, lambda_r_factor)  # noqa

    def prepare_embed(self):
        """Make embedding trainable."""
        emb = self.get_embed()
        self.embedding_original = emb[0][1].detach().clone()
        emb[0][1].requires_grad_()
        # self.embedding = emb[0][1]

    def set_trainable(self):
        """Set the data generator to train appropriate variables."""
        self.batch_grad = True

    def set_not_trainable(self):
        """Set the data generator to train appropriate variables."""
        self.batch_grad = False
        if self.embedding_grad and self.batch_grad:
            emb = self.get_embed()
            emb[0][1].requires_grad = False
            # self.embedding = emb[0][1]

    def get_embed(self):
        """Get the embedding weights."""
        return [x for x in self.named_parameters() if 'embed' in x[0]]

    def sample_batch(
            self,
            batch_size,
            force_params=None,
            force_mean=False,
            class_vector=None):
        """
        Sample a batch.

        batch_size: (int) size of batch

        Returns

        batch:  (tensor)
        labels: (tensor)

        Hold object properties constant for now across +/- samples. Fix later.
        """
        if force_params is not None:
            self.force_parameters(force_params)

        # Grab the image vector generator
        imv_gen = self.sample_lambda0_r(
            batch_size=batch_size,
            d=self.dists[0])
        if force_mean:
            noise_vector = torch.tensor(np.concatenate((
                (imv_gen.mean + 2. * imv_gen.stddev).reshape(1, -1),
                (imv_gen.mean + 1. * imv_gen.stddev).reshape(1, -1),
                (imv_gen.mean + 0. * imv_gen.stddev).reshape(1, -1),
                (imv_gen.mean + -1. * imv_gen.stddev).reshape(1, -1),
                (imv_gen.mean + -2. * imv_gen.stddev).reshape(1, -1)), 0))
        else:
            noise_vector = imv_gen.rsample([batch_size])
        noise_vector = noise_vector * self.truncation

        if FAMILY == 'mv_normal':
            # Rescale to 1 sd
            noise_vector = noise_vector / noise_vector.std()
        # Truncate to bounds
        # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        lower = (self.dist_bounds[0] - imv_gen.mean) / imv_gen.stddev
        upper = (self.dist_bounds[1] - imv_gen.mean) / imv_gen.stddev
        noise_vector = torch.max(torch.min(noise_vector, upper), lower)
        if class_vector is None:
            label_vector = np.random.randint(
                low=0, high=1000, size=[batch_size, 1])
            class_vector = np.eye(
                self.num_classes)[label_vector]
            label_vector = torch.tensor(
                label_vector.astype(np.int32),
                dtype=torch.long,
                device=self.device)
            class_vector = torch.from_numpy(
                class_vector.squeeze(1).astype(np.float32)).to(self.device)
        else:
            class_vector = torch.from_numpy(
                class_vector.astype(np.float32)).to(self.device)
            label_vector = torch.argmax(class_vector, 1)

        # Generate an image
        if self.batch_grad:
            if self.learn_truncation:
                self.truncation.requires_grad = True
            image_batch = self.model(
                noise_vector, class_vector, self.truncation)
        else:
            if self.learn_truncation:
                self.truncation.requires_grad = False
            with torch.no_grad():
                image_batch = self.model(
                    noise_vector, class_vector, self.truncation)

        # Hardcode the normalization
        image_batch = (image_batch + 1.) / 2.
        image_batch = image_batch - self.norm_mean
        image_batch = image_batch / self.norm_std

        if self.crop == 'center':
            # Hardcode a center crop
            image_batch = image_batch[..., 16:-16, 16:-16]
        return image_batch, label_vector.squeeze()


if __name__ == '__main__':
    """Run test."""
    P = Generator('biggan', 256, 'cpu', 1000)
    P._test()

