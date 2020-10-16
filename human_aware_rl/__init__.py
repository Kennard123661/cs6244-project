import os
import platform

hostname = platform.node()
if hostname == 'kennardng-K501UX':
    PROJECT_DIR = '/home/kennardng/Desktop/cs6244-project'
else:
    raise NotImplementedError
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
