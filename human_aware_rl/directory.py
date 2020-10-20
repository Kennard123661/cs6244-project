import platform
import getpass
import os


node = platform.node()
if node == 'space':
    PROJECT_DIR = '/home/space/Github/cs6244-project'
else:
    user_id = getpass.getuser()
    if user_id == 'e0036319':
        PROJECT_DIR = '/home/e/e0036319/code/cs6244-project'
    else:
        raise NotImplementedError
DATA_DIR = os.path.join(PROJECT_DIR, 'human_aware_rl', 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
