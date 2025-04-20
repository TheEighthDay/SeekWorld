import json
import os
import random
import re
from argparse import ArgumentParser
from multiprocessing import Process, Queue
import uuid
from flask import Flask, jsonify, request


import gym, Levenshtein
import gym_sokoban  
import numpy as np  
from PIL import Image  
  
def load_level_from_npy(env, level_path):  
    # Load the level data from the .npy file  
    level_data = np.load(level_path)  
      
    # Access the unwrapped environment to set the room state directly  
    env.unwrapped.room_fixed = level_data  
    env.unwrapped.room_state = np.copy(level_data)  
    env.unwrapped.player_position = np.argwhere(level_data == 5)[0]  # Assuming 5 represents the player in your saved state  
      
    # Reset the internal state based on the new room state  
    env.unwrapped.boxes_on_target = np.sum(level_data == 3)  # Assuming 3 represents a box on a target  
    env.unwrapped.num_env_steps = 0  
      
    # Recompute the initial observation  
    env.unwrapped.state = env.unwrapped.render(mode='rgb_array')  
    return env  

from gym_sokoban.envs.sokoban_env import SokobanEnv

class DirectLoadSokobanEnv(SokobanEnv):
    def __init__(self, preloaded_room):
        # 跳过原始初始化中的关卡生成代码
        super().__init__(
            preloaded_room.shape,
            max_steps=120,
            num_boxes = np.sum(preloaded_room == 4),
            reset = False
        )
        self.penalty_for_step = -0.1
        self.room_fixed = preloaded_room     # 直接设置预加载关卡
        
        self.reset()  # 仅初始化必要变量

    def reset(self):
        # 直接使用预加载关卡，跳过随机生成
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.room_state = np.copy(self.room_fixed)
        self.player_position = np.argwhere(self.room_fixed == 5)[0]
        box_positions = np.argwhere(self.room_fixed == 4)
        for p in box_positions:
            self.room_fixed[tuple(p)]=1
        self.room_fixed[tuple(self.player_position)] = 1
        # self.room_state[tuple(self.player_position)] = 1
        self.num_env_steps = 0
        return self.render("rgb_array")


def create_env_direct(level_path):
    # 提前加载关卡数据
    level_data = np.load(level_path)
    
    # 直接使用自定义环境类
    env = DirectLoadSokobanEnv(preloaded_room=level_data)
    return env


def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in index_to_path.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem


app = Flask(__name__)


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    end_think_count = content.count("</think>")
    end_answer_count = content.count("</answer>")
    all_actions = False
    try:
        acts = [act.strip().lower() for act in content.split("<answer>")[1].split("</answer>")[0].split(",")]
        all_actions = all([act in ["left", "right", "up", "down"] for act in acts])
    except:
        pass
    return [end_think_count==1, end_answer_count==1, think_count == 1, answer_count == 1, all_actions]

ACT_TO_INT = {
    "up": 1, "down": 2, "left": 3, "right": 4
}


import re

def check_repeated_substrings(text, n, max_k):
    # 预处理文本，提取单词并转换为小写
    words = re.findall(r"\b\w[\w']*\b", text.lower())
    # max_k = 10
    result = {k: False for k in range(1, max_k + 1)}
    # 初始化数据结构，每个k存储kgram信息：{kgram: (last_start, current_count)}
    k_data = {k: {} for k in range(1, max_k + 1)}
    
    for i in range(len(words)):
        for k in range(1, max_k + 1):
            # 检查是否可以形成有效的k-gram
            if i >= k - 1:
                s = i - k + 1
                if s < 0 or s + k > len(words):
                    continue
                kgram = tuple(words[s:s + k])
                
                # 更新k_data中的记录
                if kgram in k_data[k]:
                    last_start, current_count = k_data[k][kgram]
                    # 检查是否连续
                    if last_start == s - k:
                        new_count = current_count + 1
                    else:
                        new_count = 1
                else:
                    new_count = 1
                # 保存当前kgram的信息
                k_data[k][kgram] = (s, new_count)
                # 检查是否超过n次重复
                if new_count > n:
                    result[k] = True
    return any(list(result.values()))


def verify_sokoban(input_queue, output_queue):
    while True:
        task_data = input_queue.get()
        if task_data is None: 
            break
        task_id, output, env_path, env_name = task_data
        reward = 0.0
        best_reward = 0.0
        action_length_encourage = 0.0
        thinking_length_encorage = 0.0
        env = create_env_direct(env_path)
        
        # actions encorage
        acts_text = output.split("<answer>")[1].split("</answer>")[0].strip()
        acts = [act.strip().lower() for act in acts_text.split(",")]
        num_actions = len(acts)
        if len(acts) < 20:
            action_length_encourage += len(acts) * 0.05
        else:
            action_length_encourage += 20 * 0.05 - ((len(acts) - 20) /30)**2 * 0.2 
        
        
        # analyze encorage
        analyze = output.split("<think>")[1].split("</think>")[0].strip()
        ntk = list(filter(lambda x: len(x)>1, analyze.split(" ")))
        num_analyze = len(ntk)
        if len(ntk) < 200:
            thinking_length_encorage += len(ntk) * 0.001
        else:
            thinking_length_encorage += 200 * 0.001 # - ((len(ntk) - 200) /1000)**2 * 0.2  # max 1k

        # repeat penalty
        if check_repeated_substrings(acts_text, 7, 5):
            action_length_encourage = min(0, action_length_encourage - 0.2)
        # if check_repeated_substrings(analyze, 4, 50):
        #     thinking_length_encorage = min(0, thinking_length_encorage - 0.2)

        for act in acts[:50]:
            action = ACT_TO_INT[act]
            state, sreward, done, info = env.step(action)
            reward += sreward # if sreward > 0 else 0
            if reward > best_reward:
                best_reward = reward
            if done or sreward>3:
                break
            
        
        if 'env' in locals():
            del env
        output_queue.put((task_id, best_reward, action_length_encourage , thinking_length_encorage, num_actions, num_analyze))

@app.route("/get_reward", methods=["POST"])
def get_reward():
    data = request.get_json()
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    
    tasks = []
    print_tasks = dict()
    current_task_ids = []
    fm_rewards = dict()
    # 准备任务数据
    for q, problem in zip(data["query"], data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400

        # level_index = json.loads(problem)  # v0-level_0
        if problem not in index_to_path:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        
        env_path = index_to_path[problem]
        
        # level_path = index_to_path[level_index]
        env_name = ""
        for k in ["Sokoban-small-v0", "Sokoban-small-v1", "Sokoban-v0", "Sokoban-v1", "Sokoban-large-v0"]:
            if k in env_path:
                env_name = k
                break
        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        fm_details = verify_format(response)
        format_reward = float(all(fm_details))
        
        
        task_id = uuid.uuid4().hex
        current_task_ids.append(task_id)
        if format_reward > 0.1:
            fm_rewards[task_id] = format_reward
            tasks.append((task_id, response, env_path, env_name))

        do_print = random.randint(1, 20) == 1
        if do_print:
            info=f"Query: {q}\n\nProblem: {problem}\n\n Response: {response}\n\n Format Reward: {fm_details}\n\n"
            info = re.sub(r"<\|.*?\|>","",info)
            print_tasks[task_id] = info


    for task in tasks:
        input_queue.put(task)
    
    total_reward = {}
    env_reward = {}
    number_actions = {}
    number_analyze = {}
    for t in current_task_ids:
        if t not in fm_rewards:
            total_reward[t] = 0.0
            env_reward[t] = 0.0
            number_actions[t] = 0
            number_analyze[t] = 0
    while len(total_reward) < len(current_task_ids):
        task_id, task_reward, action_length_encourage , thinking_length_encorage, num_actions, num_analyze = output_queue.get()
        if task_id in print_tasks:
            print(print_tasks[task_id])
            print(f"Total reward: ", task_reward + fm_rewards[task_id] * 0.5 + action_length_encourage + thinking_length_encorage )
        if task_id in current_task_ids:
            total_reward[task_id] = task_reward + fm_rewards[task_id] * 0.5 + action_length_encourage + thinking_length_encorage
            env_reward[task_id] = task_reward
            number_actions[task_id] = num_actions
            number_analyze[task_id] = num_analyze
        else:
            # 将不属于当前请求的任务重新放回队列
            output_queue.put((task_id, task_reward, action_length_encourage , thinking_length_encorage, num_actions, num_analyze))
    
    # 按原始顺序组合结果
    total_rewards = [float(total_reward[task_id]) for task_id in current_task_ids]
    env_rewards = [float(env_reward[task_id]) for task_id in current_task_ids]
    n_actions = [float(number_actions[task_id]) for task_id in current_task_ids]
    n_analyze = [float(number_analyze[task_id]) for task_id in current_task_ids]
    return jsonify({"rewards": total_rewards, "task_rewards": env_rewards, "actions_number": n_actions, "thinking_words": n_analyze})

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Datasets to use (comma separated)", required=True
    )
    parser.add_argument(
        "--prompt-template", type=str, default=None, help="Prompt template", required=True
    )
    parser.add_argument(
        "--input_key", type=str, default="prompt", help="The key name of prompt."
    )
    args = parser.parse_args()
    
    # index_to_path = json.load(open(args.dataset))
    # Split dataset paths and load all datasets
    dataset = []
    for dataset_path in args.dataset.split(','):
        dataset_path = dataset_path.strip()
        if dataset_path.endswith("json"):
            with open(dataset_path, "r") as f:
                dataset.extend(json.load(f))
        elif dataset_path.endswith("jsonl"):
            with open(dataset_path, "r") as f:
                dataset.extend([json.loads(l) for l in f.readlines()])
        else:
            raise ValueError(f"Unsupported file format for dataset: {dataset_path}")

    format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"

    if args.prompt_template=="chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template=="qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template=="base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.dataset}")
    print("load dataset success")
    index_to_path = dict()
    for item in dataset:
        problem = item[args.input_key]
        index_to_path[problem] = item["env_path"]

    num_workers = 8
    input_queue = Queue()
    output_queue = Queue()
    
    workers = []
    for _ in range(num_workers):
        p = Process(target=verify_sokoban, args=(input_queue, output_queue))
        p.start()
        workers.append(p)
    
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    
    for _ in range(num_workers):
        input_queue.put(None) 
    for p in workers:
        p.join()
