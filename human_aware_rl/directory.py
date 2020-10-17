import platform
import os

node = platform.node()
if node == 'kennardng-desktop':
    PROJECT_DIR = '/home/kennardng/Desktop/cs6244-project'
else:
    raise NotImplementedError
DATA_DIR = os.path.join(PROJECT_DIR, 'human_aware_rl', 'data')
