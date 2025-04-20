
# To run Sokoban, you will need  
# 1. dataset.jsonl;   2. a folder containing images; 3. a folder containing game config

# Example for dataset.jsonl
# {"message": "[{\"role\": \"system\", \"content\": \"You're going to play a game of Sokoban, where the goal is to manipulate the green character to push the yellow box into the target area (an area with a red dot in the center). \\nGenerate all actions from the initial frame to the end at once.\"}, {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"file://path/to/Sokoban-v0-frame_65.png\"}, {\"type\": \"text\", \"text\": \"You should first thinks about the reasoning process in the mind and then provides the user with the answer, the answer is a long sequence of Left, Right, Up, Down, separated by ','. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, e.g., <think> To ... </think><answer> Left, Right, ... </answer>, which means your output should start with <think> and end with </answer>.\"}]}]", "question": "Sokoban-v0-level_65", "answer": "", "env_path": "path/to/Sokoban-v0-level_65.npy"}
# {"message": "[{\"role\": \"system\", \"content\": \"You're going to play a game of Sokoban, where the goal is to manipulate the green character to push the yellow box into the target area (an area with a red dot in the center). \\nGenerate all actions from the initial frame to the end at once.\"}, {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"file://path/to/Sokoban-small-v1-frame_91.png\"}, {\"type\": \"text\", \"text\": \"You should first thinks about the reasoning process in the mind and then provides the user with the answer, the answer is a long sequence of Left, Right, Up, Down, separated by ','. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, e.g., <think> To ... </think><answer> Left, Right, ... </answer>, which means your output should start with <think> and end with </answer>.\"}]}]", "question": "Sokoban-small-v1-level_91", "answer": "", "env_path": "path/to/Sokoban-small-v1-level_91.npy"}


# use examples/data/gen_sokoban_tasks.py to generate images and configs (.npy files)


export ROOT_PATH=pwd
export DATASET="/path/to/sokoban_dataset.jsonl"
wandb login "wandb key"
MODEL_CPK_NAME="ckpt name"
PRETRAIN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="/path/to/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

python -m openrlhf.models.remote_rm.sokoban_verifier --dataset $DATASET --prompt-template chatml --input_key message > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir /tmp/ray --include-dashboard=false


python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 128 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 4 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 8196 \
   --advantage_estimator gae \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 5e-6 \
   --init_kl_coef 0 \
   --lambd 1 \
   --gamma 1 \
   --prompt_data $DATASET \
   --input_key message \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 5 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --wandb_run_name $MODEL_CPK_NAME \
   --wandb_group hyper_para_search \
   --freeze_prefix visual

kill $childpid
ray stop
