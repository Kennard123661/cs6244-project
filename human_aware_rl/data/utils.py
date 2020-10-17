import pandas as pd
from ast import literal_eval
import numpy as np
import itertools

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState


def parse_action(action):
    """ parses actions """
    if isinstance(action, list):
        action = tuple(action)
    elif isinstance(action, str) and action == 'INTERACT':
        action = 'interact'
    assert action in Action.ALL_ACTIONS, 'ERR: invalid action - {}'.format(action)
    return action


def parse_joint_action(joint_action: str):
    """ parses the joint actions """
    if isinstance(joint_action, str):
        joint_action = literal_eval(joint_action)
    joint_action = tuple([parse_action(a) for a in joint_action])
    return joint_action


def parse_object_state(mdp, object_df: pd.DataFrame):
    """ parses the object state and returns the ObjectState """
    object_position = tuple(object_df['position'])
    if 'state' in object_df.keys():
        soup_type, num_items, cook_time = tuple(object_df['state'])

        # fix differing dynamics from Amazon Turk version
        if cook_time > mdp.soup_cooking_time:
            cook_time = mdp.soup_cooking_time

        object_state = (soup_type, num_items, cook_time)
    else:
        object_state = None
    return ObjectState(object_df['name'], object_position, object_state)


def parse_state(mdp, state_df: pd.DataFrame):
    """ converts the state dataframe to an Overcooked state """
    if isinstance(state_df, str):
        state_df = literal_eval(state_df)

    player_0, player_1 = state_df['players']
    position0 = tuple(player_0['position'])
    orientation0 = tuple(player_0['orientation'])
    position1 = tuple(player_1['position'])
    orientation1 = tuple(player_1['orientation'])

    held_object0, held_object1 = None, None
    if 'held_object' in player_0.keys():
        held_object0 = parse_object_state(mdp, player_0['held_object'])

    if 'held_object' in player_1.keys():
        held_object1 = parse_object_state(mdp, player_1['held_object'])

    player_state_0 = PlayerState(position=position0, orientation=orientation0, held_object=held_object0)
    player_state_1 = PlayerState(position=position1, orientation=orientation1, held_object=held_object1)

    world_objects = {}
    for world_object in state_df['objects'].values():
        object_state = parse_object_state(mdp, object_df=world_object)
        world_objects[object_state.position] = object_state

    assert not state_df["pot_explosion"]
    overcooked_state = OvercookedState(players=(player_state_0, player_state_1), objects=world_objects,
                                       order_list=None)
    return overcooked_state


def get_worker_layout_run(df: pd.DataFrame, worker_id: int, layout: str):
    """ retrieve the run data for a specified worker_id for a specific layout """
    worker_df = df[df['workerid_num'] == worker_id]
    worker_layout_df = worker_df[worker_df['layout_name'] == layout]
    return worker_layout_df


def parse_df_to_joint_trajectory(df: pd.DataFrame, check_complete_trajectory: bool = True):
    assert len(df) > 0, 'trajectory should not be empty'
    data_point = df.iloc[0]
    layout = data_point['layout_name']

    evaluator = AgentEvaluator(
        mdp_params={'layout_name': layout},
        env_params={'horizon': 1250}
    )

    env = evaluator.env
    mdp = env.mdp

    overcooked_states = [parse_state(mdp=mdp, state_df=s) for s in df.state]
    overcooked_actions = [parse_joint_action(joint_action=a) for a in df.joint_action]
    overcooked_rewards = list(df.reward)
    assert sum(overcooked_rewards) == data_point.score_total

    episode_length = len(overcooked_states)
    trajectories = {
        'ep_observations': [overcooked_states],
        'ep_actions': [overcooked_actions],
        'ep_rewards': [overcooked_rewards],

        'ep_dones': [[False] * episode_length],

        'ep_returns': [data_point.score_total],
        'ep_returns_sparse': [data_point.score_total],
        'ep_lengths': [episode_length],

        'mdp_params': [mdp.mdp_params],
        'env_params': [env.env_params],
    }

    # convert all to numpy arrays except ep-actions
    trajectories = {k: np.array(v) if k != 'ep_actions' else v for k, v in trajectories.items()}
    if check_complete_trajectory:
        evaluator.check_trajectories(trajectories=trajectories)

    trajectory_metadata = {
        'worker_id': data_point['workerid_num'],
        'round_num': data_point['round_num'],
        'mdp': evaluator.env.mdp
    }
    return trajectories, trajectory_metadata


def get_single_trajectories_from_joint_df(joint_df: pd.DataFrame, worker_ids: list, layouts: list,
                                          is_ordered_pairs: bool = True,
                                          is_human_ai_trajectories: bool = False):
    single_agent_trajectories = {
        'ep_observations': [],
        'ep_actions': [],
        'ep_rewards': [],  # individual reward values
        'ep_dones': [],

        'ep_returns': [],
        'ep_lengths': [],  # length of each episode
        'ep_agent_idxs': [],
        'mdp_params': [],
        'env_params': []
    }

    for worker_id, layout in itertools.product(worker_ids, layouts):
        single_df = get_worker_layout_run(df=joint_df, worker_id=worker_id, layout=layout)

        if len(single_df) == 0:  # no data
            print('INFO: {} layout missing from worker {}'.format(layout, worker_id))
            continue

        joint_trajectory, joint_metadata = parse_df_to_joint_trajectory(df=single_df,
                                                                        check_complete_trajectory=is_ordered_pairs)
        human_idx = [get_human_player_idx(df=single_df)] if is_human_ai_trajectories else [0, 1]

        add_single_trajectories_from_joint(single_agent_trajectories=single_agent_trajectories,
                                           joint_trajectory=joint_trajectory,
                                           joint_metadata=joint_metadata,
                                           agent_idxs=human_idx, preprocess_data=(not is_human_ai_trajectories))


def get_human_player_idx(df: pd.DataFrame):
    """Determines which player index had a human player"""
    assert len(df['workerid_num'].unique()) == 1
    return (df.groupby('workerid_num')['player_index'].sum() > 0).iloc[0]


def add_single_trajectories_from_joint(single_agent_trajectories: dict, joint_trajectory, joint_metadata,
                                       agent_idxs, preprocess_data: bool):
    """
    Takes a joint trajectory and splits it into two single-agent trajectories and adding it to the
    single_agent_trajectories dictionary.

    Args:
        single_agent_trajectories:
        joint_trajectory:
        joint_metadata:
        agent_idxs: agent idxs to convert to single trajectory.
        preprocess_data:

    Returns:

    """
    from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
    mdp = joint_metadata['mdp']
    ml_planner = MediumLevelPlanner.from_pickle_or_compute(mdp=mdp, mlp_params=NO_COUNTERS_PARAMS, force_compute=False)

    assert len(joint_trajectory['ep_observations']) == 1, 'this method takes in one trajectory'
    states = joint_trajectory['ep_observations'][0]
    joint_actions = joint_trajectory['ep_actions'][0]
    rewards = joint_trajectory['ep_rewards'][0]
    num_steps = joint_trajectory['ep_lengths'][0]

    for agent_idx in agent_idxs:
        eps_observations, eps_actions, eps_dones = [], [], []
        for i, state in enumerate(states):
            action = joint_actions[i][agent_idx]

            if preprocess_data:
                action = np.array([Action.ACTION_TO_INDEX[action]]).astype(int)
                state = mdp.featurize_state(state, ml_planner)[agent_idx]

            eps_observations.append(state)
            eps_actions.append(action)
            eps_dones.append(False)
        assert len(eps_observations) > 0, 'there should be at least one trajectory'
        eps_dones[-1] = True

        single_agent_trajectories['ep_observations'].append(eps_observations)
        single_agent_trajectories['ep_actions'].append(eps_actions)
        single_agent_trajectories['ep_dones'].append(eps_dones)
        single_agent_trajectories['ep_rewards'].append(rewards)
        single_agent_trajectories['ep_returns'].append(sum(rewards))
        single_agent_trajectories['ep_lengths'].append(num_steps)
        single_agent_trajectories['ep_agent_idxs'].append(agent_idx)
        single_agent_trajectories['mdp_params'].append(mdp.mdp_params)
        single_agent_trajectories['env_params'].append({})


if __name__ == '__main__':
    pass
