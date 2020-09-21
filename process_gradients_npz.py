import os
import numpy as np
import pandas as pd
from glob import glob
from utils import normalize_weights
from db import db


pull_from_db = True
if pull_from_db:
    files = db.get_gradients_info()
else:
    results_dir = 'results'
    wc = 'gradients.npz'
    file_path = os.path.join(results_dir, '*{}'.format(wc))
    files = glob(file_path)
models, res = [], []
for f in files:
    try:
        if pull_from_db:
            data = np.load('{}.npz'.format(f['file_path']))
        else:
            data = np.load(f)
        model_name = data['model_name']
        siamese = data['siamese']
        dataset = data['dataset']
        inner_steps = data['inner_steps']
        # Using weight norm?
        wn = data['wn']
        if wn:
            num_objects_center = normalize_weights(data, 'num_objects', 'center')  # noqa
            num_objects_scale = normalize_weights(data, 'num_objects', 'scale')  # noqa
            object_size_center = normalize_weights(data, 'object_size', 'center')  # noqa
            object_size_scale = normalize_weights(data, 'object_size', 'scale')  # noqa
            object_location_center = normalize_weights(data, 'object_location', 'center')  # noqa
            object_location_scale = normalize_weights(data, 'object_location', 'scale')  # noqa
        else:
            num_objects_center = data['num_objects_center']
            num_objects_scale = data['num_objects_scale']
            object_size_center = data['object_size_center']
            object_size_scale = data['object_size_scale']
            object_location_center = data['object_location_center']
            object_location_scale = data['object_location_scale']
        f_vect = [model_name, siamese, dataset, inner_steps, wn]  # noqa
        models += [f_vect]
        f_vect = [num_objects_center, num_objects_scale, object_size_center, object_size_scale, num_objects_center, num_objects_scale, object_size_center, object_size_scale, object_location_center, object_location_scale]  # noqa
        res += [f_vect]
    except Exception as e:
        print('Failed to load {}, {}'.format(f, e))

df = pd.DataFrame(models, columns=['model_name', 'siamese', 'dataset', 'inner_steps', 'wn'])  # noqa
data_df = pd.DataFrame(res, columns=['num_objects_center', 'num_objects_scale', 'object_size_center', 'object_size_scale', 'num_objects_center', 'num_objects_scale', 'object_size_center', 'object_size_scale', 'object_location_center', 'object_location_scale'])  # noqa
