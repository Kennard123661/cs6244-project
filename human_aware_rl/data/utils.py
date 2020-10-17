import pandas as pd
import pickle
from ast import literal_eval
import numpy as np
import itertools
import os
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action

from human_aware_rl import DATA_DIR


def get_trajectories_from_data(data_file: str, layouts: list):
    print('INFO: loading data from {}'.format(data_file))
    data = pd.read_pickle(data_file)
    data.to_csv(data_file[:-4] + '.csv')
    convert_df_to_single_trajectories(df=data, layouts=layouts)


def convert_df_to_single_trajectories(df: pd.DataFrame, layouts: list, is_ordered_pairs: bool = True,
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

    worker_ids = np.unique(df['workerid_num'])
    for worker_id, layout in itertools.product(worker_ids, layouts):
        single_df = get_worker_layout_data(all_df=df, worker_id=worker_id, layout=layout)

        if len(single_df) == 0:  # no data
            print('INFO: {} layout missing from worker {}'.format(layout, worker_id))
            continue

        joint_trajectory, joint_metadata = convert_df_to_joint_trajectory(df=single_df)
        human_idx = [get_human_player_idx(df=single_df) if is_human_ai_trajectories else [0, 1]]


def convert_joint_trajectories_to_single(single_agent_trajectories, joint_trajectory, joint_metadata, human_idxs,
                                         is_processed: bool):
    env = joint_trajectory


def get_human_player_idx(df: pd.DataFrame):
    human_player_idxs = []
    assert len(one_traj_df['player_0_id'].unique()) == 1
    assert len(one_traj_df['player_1_id'].unique()) == 1


def parse_state(state: str) -> dict:
    state = literal_eval(state)
    return state


def parse_action(action):
    if isinstance(action, list):
        action = tuple(action)
    elif isinstance(action, str):
        action = str(action).lower()
    assert action in Action.ALL_ACTIONS, '{}'.format(action)
    return action


def parse_joint_action(joint_action: str):
    joint_action = literal_eval(joint_action)
    joint_action = tuple([parse_action(a) for a in joint_action])
    return joint_action


def convert_df_to_joint_trajectory(df: pd.DataFrame, check_complete_trajectory: bool = True):
    assert len(df) > 0, 'trajectory should not be empty'
    data_point = df.iloc[0]
    layout = data_point['layout_name']

    evaluator = AgentEvaluator.from_layout_name(
        mdp_params={'layout_name': layout},
        env_params={'horizon': 1250}
    )

    env = evaluator.env
    mdp = env.mdp

    overcooked_states = [parse_state(s) for s in df.state]
    overcooked_actions = [parse_joint_action(a) for a in df.joint_action]
    overcooked_rewards = list(df.reward)
    assert sum(overcooked_rewards) == data_point.score_total

    num_steps = len(overcooked_states)
    trajectories = {
        'eps_observations': [overcooked_states],
        'eps_actions': [overcooked_actions],
        'eps_rewards': [overcooked_rewards],

        'eps_dones': [[False] * num_steps],
        'eps_infos': [{}] * num_steps,

        'eps_returns': [data_point.score_total],
        'eps_lengths': [num_steps],
        'mdp_params': [mdp.mdp_params],
        'env_params': [env.env_params],
    }

    # convert all to numpy arrays except ep-actions
    trajectories = {k: np.array(v) if k != 'ep-actions' else v for k, v in trajectories.items()}
    if check_complete_trajectory:
        evaluator.check_trajectories(trajectories=trajectories)

    metadata = {
        'worker_id': data_point['workerid_num'],
        'round_num': data_point['round_num'],
        'mdp': evaluator.env.mdp
    }
    return trajectories, metadata



def get_worker_layout_data(all_df: pd.DataFrame, worker_id: int, layout: str):
    """
    Args:
        all_df: the complete data frame
        worker_id: worker id to extract the runs for
        layout: layout ot extract the runs for

    Returns:
        data corresponding to the worker id for the specific layout
    """
    worker_df = all_df[all_df['workerid_num'] == worker_id]
    worker_layout_df = worker_df[worker_df['layout_name'] == layout]
    return worker_layout_df


def main():
    data_dir = os.path.join(DATA_DIR, 'human', 'anonymized')
    train_data_file = os.path.join(data_dir, 'clean_train_trials.pkl')
    layouts = ['asymmetric_advantages']
    get_trajectories_from_data(data_file=train_data_file, layouts=layouts)


if __name__ == '__main__':
    main()



