copy folders in human_aware_rl/data to checkpoints
cd into human_aware_rl
./experiments/ppo_bc_experiments.sh

python ppo/ppo.py with EX_NAME="ppo_bc_train_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=False > temp.log
python ppo/ppo.py with EX_NAME="ppo_bc_train_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456]" VF_COEF=0.5 MINIBATCHES=1 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e2, 3e3]" TIMESTAMP_DIR=False
