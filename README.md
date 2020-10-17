# CS6244 Project

## Setup

This is the code to setup the repository

```
conda create -n cs6244-project python==3.7
conda activate cs6244 project
pip install -r requirements.txt
./install.sh
```

Then, go to `human_aware_rl/directory.py` and update the following paths to your dataset:
```python
import platform
import os

node = platform.node()
if node == 'kennardng-desktop':
    PROJECT_DIR = '/home/kennardng/Desktop/cs6244-project'
elif node == '<YOUR HOSTNAME>':
    PROJECT_DIR = '<YOUR PROJECT DIRECTORY'
else:
    raise NotImplementedError
DATA_DIR = os.path.join(PROJECT_DIR, 'human_aware_rl', 'data')
```
where you should update `<YOUR HOSTNAME>` and `<YOUR PROJECT DIRECTORY>` accordingly. To test if you setup correctly,
try running `python human_aware_rl/experiments/bc_experiments.py`
