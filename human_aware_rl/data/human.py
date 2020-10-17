import pandas as pd
import pickle
import os

from human_aware_rl import DATA_DIR


def get_trajectories_from_data(df: pd.DataFrame, layouts: list, is_ordered_pairs: bool = True,
                               is_human_ai_trajectories: bool = False):
    single_agent_trajectories = {
        'eps-observations': [],
        'eps-actions': [],
        'eps-rewards': [],  # individual reward values
        'eps-dones': [],

        'eps-returns': [],
        'eps-lengths': [],  # length of each episode
        'eps-agent-idxs': [],
        'mdp-params': [],
        'env-params': []
    }

    worker_ids = df['worker']


def main():
    data_dir = os.path.join(DATA_DIR, 'human', 'anonymized')
    train_data_file = os.path.join(data_dir, 'clean_train_trials.pkl')
    get_trajectories_from_data(data_file=train_data_file, layouts=None)


if __name__ == '__main__':
    main()



