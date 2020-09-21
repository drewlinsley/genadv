'''DB tools'''
import os
import yaml
import logging
import time
import argparse
import datetime
import itertools
# import numpy as np
# import pandas as pd
from db import db
from config import Config


main_config = Config()
dt = datetime.datetime.fromtimestamp(
    time.time()).strftime('%Y-%m-%d_%H:%M:%S')
logging.basicConfig(
    filename=os.path.join(
        main_config.db_log_files,
        '{}.log'.format(dt)),
    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger()


def main(
        init_db=False,
        repeat=0,
        reset_exps=False,
        reset_grads=False,
        add_to_exp_db=True,
        add_experiment=None):
    """Routines for adjusting the DB."""
    if init_db:
        # Create the DB from a schema file
        db.initialize_database()
        logger.info('Initialized database.')

    if reset_exps:
        db.reset_experiments()

    if reset_grads:
        db.reset_gradients()

    if add_experiment is not None:
        with open(add_experiment) as f:
            exp = yaml.load(f, Loader=yaml.FullLoader)

        # Generate experiment combinations
        exps = list(itertools.product(*exp.values()))
        logger.info('Adding the following experiments to the DB.')
        logger.info(exp.keys())
        logger.info(exps)

        # Tag each exps with dt
        exp_keys = list(exp.keys()) + ['dt', 'is_processing', 'finished']

        # Create list of dicts for adding to the DB
        exp_dicts = []
        for row in exps:
            row = list(row)
            row += (dt, False, False,)  # is_processing, finished
            d = {k: v for k, v in zip(exp_keys, row)}
            exp_dicts += [d]
        for _ in range(repeat):
            exp_dicts += exp_dicts
        logger.info('Adding {} total experiments.'.format(len(exp_dicts)))
        if add_to_exp_db:
            db.add_experiments(exp_dicts, use_exps=True)
        else:
            db.add_experiments(exp_dicts, use_exps=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init_db',
        dest='init_db',
        action='store_true',
        help='Initialize DB from schema (must be run at least once).')
    parser.add_argument(
        '--reset_exps',
        dest='reset_exps',
        action='store_true',
        help='Reset the experiment table.')
    parser.add_argument(
        '--reset_grads',
        dest='reset_grads',
        action='store_true',
        help='Reset the gradient table.')
    parser.add_argument(
        '--grads',
        dest='add_to_exp_db',
        action='store_false',
        help='Add to gradient DB.')
    parser.add_argument(
        '--exp',
        dest='add_experiment',
        type=str,
        default=None,
        help='Add an experiment')
    parser.add_argument(
        '--repeat',
        dest='repeat',
        type=int,
        default=0,
        help='Repeat experiment N times')
    args = parser.parse_args()
    main(**vars(args))
