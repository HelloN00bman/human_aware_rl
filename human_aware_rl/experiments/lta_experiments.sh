python ../lta_ppo/train_model_on_bc_only.py with EX_NAME="testing_lta" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=False
