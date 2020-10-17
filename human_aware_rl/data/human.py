import pandas as pd
import pickle
import os


from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import mean_and_std_err

from human_aware_rl import DATA_DIR
from human_aware_rl.data.utils import get_single_trajectories_from_joint_df, get_worker_layout_run, \
    parse_df_to_joint_trajectory


def get_trajectories(data_file: str, layouts: list, is_ordered_trajectories: bool, is_human_ai_trajectories: bool):
    """ loads trajectories from the data file and returns single trajectories """
    assert os.path.exists(data_file), 'ERR: {} does not exist'.format(data_file)
    print('INFO: loading data from {}'.format(data_file))
    df = pd.read_pickle(data_file)

    worker_ids = list(df['workerid_num'].unique())

    trajectories = get_single_trajectories_from_joint_df(joint_df=df, worker_ids=worker_ids, layouts=layouts,
                                                         is_ordered_pairs=is_ordered_trajectories,
                                                         is_human_ai_trajectories=is_human_ai_trajectories)
    return trajectories


def get_overcooked_trajectory_from_worker_layout(df: pd.DataFrame, worker_id: int, layout: str,
                                                 check_complete_trajectory: bool = True):
    trajectory_df = get_worker_layout_run(df=df, worker_id=worker_id, layout=layout)
    trajectory, metadata = parse_df_to_joint_trajectory(df=trajectory_df,
                                                        check_complete_trajectory=check_complete_trajectory)
    if trajectory is None:
        print('WARNING: layout {} missing from worker {}'.format(layout, worker_id))
    return trajectory, metadata


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



