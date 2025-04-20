#!/bin/bash
# =================== User Configuration ===================
# Please modify these variables according to your environment
# =========================================================

# Base paths - MODIFY THESE
export WORKSPACE_DIR="$(pwd)"                      # Path to project root directory
export DATASET_PATH="./data/VerMulti/mathv_geo_message.jsonl"  # Path to your dataset
export PRETRAIN_MODEL_PATH="./checkpoints/lmm-r1-fre-text/"  # Path to pretrained model
export SAVE_PATH="/checkpoints"                   # Absolute path to save checkpoints

# Model configuration
export MODEL_NAME="lmm-r1-mgt-geo"              # Name for this training run

# Wandb configuration (optional)
export WANDB_DIR="${WORKSPACE_DIR}"                # Directory for wandb files
export WANDB_API_KEY="YOUR_WANDB_API_KEY"          # Your wandb API key (if online)

# =================== Script Execution ===================
# You shouldn't need to modify anything below this line
# ======================================================

# Get script PID and setup directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Stop any existing ray processes
ray stop

# Create necessary directories
mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"

# Print help information
echo "================================================================"
echo "LMM-R1 MGT Geometry Training"
echo "================================================================"
echo "Model name: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
echo "Logs will be saved to: ${CUR_LOG_DIR}"
echo
echo "To monitor logs:"
echo "  tail -f ${CUR_LOG_DIR}/train.log"
echo
echo "================================================================"

# Start ray
echo "Starting ray..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir ~/.cache/ray

# Start remote reward model server
echo "Starting remote reward model server..."
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset "${DATASET_PATH}" \
    --input_key message \
    --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
REMOTE_RM_PID=$!

# Start training
echo "Starting training..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
   -- python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain ${PRETRAIN_MODEL_PATH} \
   --save_path ${SAVE_PATH}/${MODEL_NAME} \
   --micro_train_batch_size 2 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 256 \
   --temperature 1.0 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --advantage_estimator gae \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.001 \
   --prompt_data ${DATASET_PATH} \
   --input_key message \
   --normalize_reward \
   --flash_attn \
   --lambd 1 \
   --gamma 1 \
   --gradient_checkpointing \
   --save_steps 20 \
   --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
   --save_hf_ckpt \
   --load_checkpoint \
   --use_wandb ${WANDB_API_KEY} \
   --wandb_run_name ${MODEL_NAME} \
   --wandb_group "lmm-r1-training" \
   --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &

TRAIN_PID=$!

# Record process IDs
echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

# Wait for training to complete
echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"

# Uncomment to wait for training to complete before exiting
# wait $TRAIN_PID

# Cleanup instructions
echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "ray stop"
echo "All logs are available in ${CUR_LOG_DIR}"