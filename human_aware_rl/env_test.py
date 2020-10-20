from human_aware_rl.baselines_utils import get_vectorized_gym_env
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from baselines_utils import RewardShapingEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym


def main():
    mdp_params = {
                    'layout_name': 'simple',
                    'start_order_list': None,
                    'rew_shaping_params': {'PLACEMENT_IN_POT_REW': 3, 'DISH_PICKUP_REWARD': 3, 'SOUP_PICKUP_REWARD': 5, 'DISH_DISP_DISTANCE_REW': 0, 'POT_DISTANCE_REW': 0, 'SOUP_DISTANCE_REW': 0}
                 }

    env_params = {'horizon': 400}
    params = {
        'mdp_params': mdp_params,
        'env_params': env_params,
        'sim_threads': 1,
        'RUN_TYPE': "ppo"
    }

    mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
    env = OvercookedEnv(mdp, **params["env_params"])
    featurize_fn=lambda x: mdp.lossless_state_encoding(x)

    def gym_env_fn():
        gym_env = gym.make('Overcooked-v0')
        if params["RUN_TYPE"] == "joint_ppo":
            # If doing joint training, action space will be different (^2 compared to single agent training)
            gym_env.custom_init(env, joint_actions=True,
                                featurize_fn=featurize_fn, baselines=True)
        else:
            gym_env.custom_init(
                env, featurize_fn=featurize_fn, baselines=True)
        return gym_env
    single_env = gym_env_fn()

    # box (self.low.min(), self.high.max(), self.shape, self.dtype)
    # shape (5, 4, 20)
    print(single_env.observation_space)
    # vectorized_gym_env = RewardShapingEnv(
    #     SubprocVecEnv([gym_env_fn] * params["sim_threads"]))
    # gym_env.self_play_randomization = 0 if params["SELF_PLAY_HORIZON"] is None else 1
    # gym_env.trajectory_sp = True #params["TRAJECTORY_SELF_PLAY"]

if __name__ == '__main__':
    main()
