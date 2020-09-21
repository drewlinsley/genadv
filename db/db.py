#!/usr/bin/env python
import os
import json
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
# import numpy as np
# from tqdm import tqdm
from db import credentials
from config import Config


main_config = Config()
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread


class db(object):
    def __init__(self, config):
        """Init global variables."""
        self.status_message = False
        self.db_schema_file = os.path.join('db', 'db_schema.txt')
        # Pass config -> this class
        for k, v in list(config.items()):
            setattr(self, k, v)

    def __enter__(self):
        """Enter method."""
        try:
            if main_config.db_ssh_forward:
                forward = sshtunnel.SSHTunnelForwarder(
                    credentials.machine_credentials()['ssh_address'],
                    ssh_username=credentials.machine_credentials()['username'],
                    ssh_password=credentials.machine_credentials()['password'],
                    remote_bind_address=('127.0.0.1', 5432))
                forward.start()
                self.forward = forward
                self.pgsql_port = forward.local_bind_port
            else:
                self.forward = None
                self.pgsql_port = ''
            pgsql_string = credentials.postgresql_connection(
                str(self.pgsql_port))
            self.pgsql_string = pgsql_string
            self.conn = psycopg2.connect(**pgsql_string)
            self.conn.set_isolation_level(
                psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            self.cur = self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor)
        except Exception as e:
            self.close_db()
            if main_config.db_ssh_forward:
                self.forward.close()
            print(e)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        if exc_type is not None:
            print((exc_type, exc_value, traceback))
            self.close_db(commit=False)
        else:
            self.close_db()
        if main_config.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        """Commit changes and exit the DB."""
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def recreate_db(self):
        """Initialize the DB from the schema file."""
        db_schema = open(self.db_schema_file).read().splitlines()
        for s in db_schema:
            t = s.strip()
            if len(t):
                self.cur.execute(t)

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print(('Successful %s.' % label))
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                )

    def DEP_reset_experiments(self):
        """Reset in coordinate info."""
        self.cur.execute(
            """
            UPDATE
            experiments SET
            is_processing=False, finished=False
            """
        )
        if self.status_message:
            self.return_status('RESET')

    def DEP_reset_gradients(self):
        """Reset in coordinate info."""
        self.cur.execute(
            """
            UPDATE
            gradients SET
            is_processing=False, finished=False
            """
        )
        if self.status_message:
            self.return_status('RESET')

    def DEP_reset_experiments(self):
        """Reset in coordinate info."""
        self.cur.execute(
            """
            DELETE FROM experiments
            """
        )
        if self.status_message:
            self.return_status('RESET')

    def reset_gradients(self):
        """Reset in coordinate info."""
        self.cur.execute(
            """
            DELETE FROM gradients
            """
        )
        if self.status_message:
            self.return_status('RESET')

    def get_grad_info(self):
        """Get gradient data."""
        self.cur.execute(
            """
            SELECT * FROM gradients where finished=True
            """
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def update_grad(self, namedict):
        """Update grad experiment row with _id."""
        self.cur.executemany(
            """
            UPDATE gradients
            SET is_processing=True, finished=TRUE, file_path=%(file_path)s
            WHERE _id=%(_id)s
            """,
            namedict)
        if self.status_message:
            self.return_status('UPDATE')

    def update_exp(self, namedict):
        """Update experiment row with _id."""
        self.cur.executemany(
            """
            UPDATE experiments
            SET is_processing=True, finished=TRUE
            WHERE _id=%(_id)s
            """,
            namedict)
        if self.status_message:
            self.return_status('UPDATE')

    def add_experiments_to_db(self, namedict):
        """
        Add a combination of parameter_dict to the db.
        ::
        """
        self.cur.executemany(
            """
            INSERT INTO experiments
            (
                dataset,
                inner_lr,
                outer_lr,
                inner_steps,
                inner_steps_first,
                outer_steps,
                epochs,
                alpha,
                beta,
                outer_batch_size_multiplier,
                model_name,
                optimizer,
                batch_size,
                adv_version,
                siamese,
                siamese_version,
                pretrained,
                amsgrad,
                task,
                wn,
                dt,
                gen_tb,
                loss,
                save_i_params,
                inner_loop_criterion,
                outer_loop_criterion,
                inner_loop_nonfirst_criterion,
                inner_lr_first,
                inner_lr_nonfirst,
                regularization
            )
            VALUES
            (
                %(dataset)s,
                %(inner_lr)s,
                %(outer_lr)s,
                %(inner_steps)s,
                %(inner_steps_first)s,
                %(outer_steps)s,
                %(epochs)s,
                %(alpha)s,
                %(beta)s,
                %(outer_batch_size_multiplier)s,
                %(model_name)s,
                %(optimizer)s,
                %(batch_size)s,
                %(adv_version)s,
                %(siamese)s,
                %(siamese_version)s,
                %(pretrained)s,
                %(amsgrad)s,
                %(task)s,
                %(wn)s,
                %(dt)s,
                %(gen_tb)s,
                %(loss)s,
                %(save_i_params)s,
                %(inner_loop_criterion)s,
                %(outer_loop_criterion)s,
                %(inner_loop_nonfirst_criterion)s,
                %(inner_lr_first)s,
                %(inner_lr_nonfirst)s,
                %(regularization)s
            )
            ON CONFLICT DO NOTHING""",
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def add_gradients_to_db(self, namedict):
        """
        Add a combination of parameter_dict to the db.
        ::
        """
        self.cur.executemany(
            """
            INSERT INTO gradients
            (
                dataset,
                inner_lr,
                outer_lr,
                inner_steps,
                outer_steps,
                epochs,
                alpha,
                beta,
                outer_batch_size_multiplier,
                model_name,
                optimizer,
                batch_size,
                adv_version,
                siamese,
                siamese_version,
                pretrained,
                amsgrad,
                task,
                wn,
                dt,
                loss,
                inner_loop_criterion,
                outer_loop_criterion,
                inner_loop_nonfirst_criterion
            )
            VALUES
            (
                %(dataset)s,
                %(inner_lr)s,
                %(outer_lr)s,
                %(inner_steps)s,
                %(outer_steps)s,
                %(epochs)s,
                %(alpha)s,
                %(beta)s,
                %(outer_batch_size_multiplier)s,
                %(model_name)s,
                %(optimizer)s,
                %(batch_size)s,
                %(adv_version)s,
                %(siamese)s,
                %(siamese_version)s,
                %(pretrained)s,
                %(amsgrad)s,
                %(task)s,
                %(wn)s,
                %(dt)s,
                %(loss)s,
                %(inner_loop_criterion)s,
                %(outer_loop_criterion)s,
                %(inner_loop_nonfirst_criterion)s
            )
            ON CONFLICT DO NOTHING""",
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def get_results(self):
        """Return results."""
        self.cur.execute(
            """
            SELECT *
            FROM experiments
            LEFT JOIN results
            ON experiments._id=results.experiment_id
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def get_trial(self):
        """Set is_processing=True."""
        self.cur.execute(
            """
            UPDATE experiments
            SET is_processing=True
            WHERE _id
            IN (SELECT _id FROM experiments WHERE is_processing=False ORDER BY _id LIMIT 1)
            RETURNING *
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_grad_trial(self):
        """Set is_processing=True."""
        self.cur.execute(
            """
            UPDATE gradients
            SET is_processing=True
            WHERE _id
            IN (SELECT _id FROM gradients WHERE is_processing=False ORDER BY _id LIMIT 1)
            RETURNING *
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def add_results_to_db(self, namedict):
        """Set is_processing=True."""
        self.cur.executemany(
            """
            INSERT INTO results
            (
                experiment_id,
                inner_loss,
                outer_loss,
                inner_loop_steps,
                outer_loop_steps,
                net_loss,
                params
            )
            VALUES
            (
                %(experiment_id)s,
                %(inner_loss)s,
                %(outer_loss)s,
                %(inner_loop_steps)s,
                %(outer_loop_steps)s,
                %(net_loss)s,
                %(params)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def add_meta_to_results(self, namedict):
        """Set is_processing=True."""
        self.cur.executemany(
            """
            INSERT INTO meta_results
            (
                experiment_id,
                meta
            )
            VALUES
            (
                %(experiment_id)s,
                %(meta)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')


def initialize_database():
    """Initialize and recreate the database."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.recreate_db()
        db_conn.return_status('CREATE')


def reset_experiments():
    """Reset experiment progress."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_experiments()
        db_conn.return_status('UPDATE')


def reset_gradients():
    """Reset gradient progress."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_gradients()
        db_conn.return_status('UPDATE')


def add_experiments(exps, use_exps=True):
    """Add coordinates to DB."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        if use_exps:
            db_conn.add_experiments_to_db(exps)
        else:
            db_conn.add_gradients_to_db(exps)
        db_conn.return_status('CREATE')


def add_results(exps):
    """Add results to DB."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.add_results_to_db(exps)
        db_conn.return_status('INSERT')
    exps = [{'_id': exps[0]['experiment_id']}]
    update_experiment(exps)


def add_meta_to_results(exp_id, meta):
    """Add experiment meta info to results table."""
    meta = json.dumps(meta)
    res = [{
        'experiment_id': exp_id,
        'meta': meta
    }]
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.add_meta_to_results(res)


def get_experiment_trial(use_exps=True):
    """Pull and reserve an experiment trial."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        if use_exps:
            trial = db_conn.get_trial()
        else:
            trial = db_conn.get_grad_trial()
        db_conn.return_status('SELECT')
    return trial


def get_gradients_info():
    """Pull and reserve an experiment trial."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        results = db_conn.get_grad_info()
        db_conn.return_status('SELECT')
    return results


def get_results():
    """Gather all results."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        results = db_conn.get_results()
        db_conn.return_status('SELECT')
    return results


def update_grad_experiment(exps):
    """Set gradient experiment to finished."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.update_grad(exps)
        db_conn.return_status('UPDATE')


def update_experiment(exps):
    """Set experiment to finished."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.update_exp(exps)
        db_conn.return_status('UPDATE')


def main(
        initialize_db):
    """Test the DB."""
    if initialize_db:
        print('Initializing database.')
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))
