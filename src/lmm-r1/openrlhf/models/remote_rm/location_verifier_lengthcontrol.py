import json
import os
import random
import re
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import Levenshtein
from flask import Flask, jsonify, request
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from loguru import logger
from concurrent import futures
app = Flask(__name__)

problem_to_answer = {}


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "", "<|endoftext|>"]
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
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1 



def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem


def verify_math(content,sol):
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            reward = 1.0
            print("Failed to verify: ", e)
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", sol)
    return reward


def standardize_region_name(region):
    """
    标准化地区名称，移除常见的后缀
    """
    region = region.lower().strip()
    # 移除常见的后缀
    suffixes = [
        ' province', ' sheng', ' län', ' oblast', ' governorate',' city',
        ' province', ' state', ' territory', ' region', ' district'
    ]
    for suffix in suffixes:
        if region.endswith(suffix):
            region = region[:-len(suffix)]
    return region.strip()

def compare_region_words(pred_words, true_words):
    """
    比较两个地区名称的单词是否匹配
    使用 Levenshtein 距离，如果两个单词的距离小于3，则认为匹配(有些地区因为发音不一样，存在1-2个字符差异)
    """
    if not pred_words or not true_words:
        return False
        
    # 记录已匹配的单词，避免重复匹配
    matched_true_words = set()
    
    # 对每个预测的单词
    for pred_word in pred_words:
        found_match = False
        # 在真实单词中寻找匹配
        for i, true_word in enumerate(true_words):
            if i in matched_true_words:
                continue
            # 如果 Levenshtein 距离小于3，认为匹配
            if Levenshtein.distance(pred_word, true_word) < 3:
                matched_true_words.add(i)
                found_match = True
                break
        # 如果当前预测单词没有找到匹配，返回False
        if not found_match:
            return False
            
    # 检查是否所有真实单词都被匹配
    return len(matched_true_words) == len(true_words)

def verify_location(content, sol):
    """
    验证位置信息的正确性
    Args:
        content: 模型的输出，格式为 <think>...</think><answer>$country,region$</answer>
        sol: 真实答案，格式为 $country,region1/region2/region3$，其中 region1/region2/region3 是同一个地区的不同别名
    Returns:
        float: 1.0 如果完全匹配，0.0 如果国家匹配但地区不匹配，0.0 如果国家不匹配
    """
    # 提取答案部分
    answer_match = re.search(r"<answer>\$(.*?)\$</answer>", content)
    if not answer_match:
        return 0.0
    
    # 清理答案中的空格
    answer = answer_match.group(1).strip()
    sol = sol.strip().strip('$').strip()
    
    # 分割国家和行政区
    try:
        pred_country, pred_region = [x.strip().lower() for x in answer.split(',')]
        true_country, true_regions = [x.strip().lower() for x in sol.split(',')]
    except ValueError:
        return 0.0
    
    # 国家必须完全匹配
    if pred_country != true_country:
        return 0.0
    
    # 标准化预测的地区名称
    pred_region = standardize_region_name(pred_region)
    
    # 将预测的地区名称分割成单词
    pred_words = [x.strip() for x in pred_region.split()]
    
    # 检查预测的地区是否匹配任何一个真实地区别名
    for true_region in true_regions.split('/'):
        true_region = standardize_region_name(true_region)
        true_words = [x.strip() for x in true_region.split()]
        
        # 如果单词匹配，返回1.0
        if compare_region_words(pred_words, true_words):
            return 1.0
    
    return 0.0

def get_think_length_reward(content):
    """
    根据思考部分的长度计算奖励分数，
    去除空格、回车、制表符等无意义字符后计算实际长度
    """
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if not think_match:
        return 0.0
        
    think_content = think_match.group(1)
    # 移除所有空白字符（空格、制表符、换行符等）和标点符号
    clean_content = re.sub(r'[\s\.,，。！？\!\?:：;；\(\)（）\[\]【】\{\}]+', '', think_content)
    length = len(clean_content)

    reward = length * 0.001
    reward = min(reward, 1.0) #max 1000字
    
    return reward

@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    rewards = []
    format_rewards = []
    acc_rewards_futures = []
    think_length_rewards = []
    
    for q,problem in zip(data["query"],data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        answer = problem_to_answer[problem]
        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
            
        format_reward = float(verify_format(response)) * 0.5
        acc_reward_future = math_verify_executor.submit(verify_location, response, answer)
        think_length_reward = get_think_length_reward(response)
       
        do_print = random.randint(1, 20) == 1
        if do_print:
            info=f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward_future.result()}\n\n Think Length Reward: {think_length_reward}\n\n"
            info = re.sub(r"<\|.*?\|>","",info)
            logger.info(info)
            
        format_rewards.append(format_reward)
        acc_rewards_futures.append(acc_reward_future)
        think_length_rewards.append(think_length_reward)
        
    acc_rewards = [f.result() for f in acc_rewards_futures]
    rewards = [f + a + t for f, a, t in zip(format_rewards, acc_rewards, think_length_rewards)]
    # 只对于非0的acc_rewards 进行length的惩罚与激励
    # 返回包含 rewards 的响应
    return jsonify({
        "rewards": rewards,
        "format_rewards": format_rewards,
        "acc_rewards": acc_rewards,
        "think_length_rewards": think_length_rewards
    })


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
    parser.add_argument("--log_file", type=str, default="remote_detailed_rm.log", help="Log file path")
    args = parser.parse_args()
    logger.remove()
    logger.add(args.log_file)
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

    format_pattern = r"^<think>(?:(?!</think>).)*</think>\s*<answer>(?:(?!</answer>).)*</answer>$"

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
    for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"].strip()
        # we require the answer to be in latex format
        if answer[0] != "$":
            answer = "$" + answer + "$"
        problem_to_answer[problem] = answer

    # math_verify can only run in main thread
    math_verify_executor = futures.ProcessPoolExecutor(max_workers=16)

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    math_verify_executor.shutdown()