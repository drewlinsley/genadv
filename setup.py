import os
import logging
# from setuptools import setup
import time
import datetime
from db import credentials
from config import Config


main_config = Config()
dt = datetime.datetime.fromtimestamp(
    time.time()).strftime('%Y-%m-%d_%H:%M:%S')
logging.basicConfig(
    filename=os.path.join(
        main_config.db_log_files,
        'setup_{}.log'.format(dt)), level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger()


"""e.g. python setup.py install"""
try:
    from pip.req import parse_requirements
    install_reqs = parse_requirements('requirements.txt', session='hack')
    # reqs is a list of requirement
    # e.g. ['django==1.5.1', 'mezzanine==1.4.6']
    reqs = [str(ir.req) for ir in install_reqs]
except Exception:
    print('Failed to import parse_requirements.')


# try:
#     setup(
#         name="latent_adv",
#         version="0.0.1",
#         packages=install_reqs,
#     )
# except Exception as e:
#     print(
#         'Failed to install requirements and compile repo. '
#         'Try manually installing. %s' % e)

config = Config()
logger.info('Installed required packages and created paths.')

params = credentials.postgresql_connection()
sys_password = credentials.machine_credentials()['password']
os.popen(
    'sudo -u postgres createuser -sdlP %s' % params['user'], 'w').write(
    sys_password)
os.popen(
    'sudo -u postgres createdb %s -O %s' % (
        params['database'],
        params['user']), 'w').write(sys_password)
logger.info('Created DB.')
