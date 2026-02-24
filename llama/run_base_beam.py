import os
import json
import argparse
import datetime
import random
from tqdm import tqdm
import torch

from src.ours.tasks import get_task
from src.ours.methods.bfs_base import solve
from src.ours.models_llama import Llama, set_configs

# sh scripts/{game24, text, crosswords}/{standard_sampling, cot_sampling, bfs}.sh
# sh scripts/game24/beam_search.sh

import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def inspect_and_convert(obj, depth=0):
    """Recursively inspect types and convert ndarray to list."""
    indent = "  " * depth  # Indentation for better readability
    if isinstance(obj, dict):
        # print(f"{indent}Inspecting dict:")
        for key, value in obj.items():
            # print(f"{indent}- Key: {key}, Type: {type(value)}")
            obj[key] = inspect_and_convert(value, depth + 1)
    elif isinstance(obj, list):
        # print(f"{indent}Inspecting list:")
        obj = [inspect_and_convert(item, depth + 1) for item in obj]
    elif isinstance(obj, np.ndarray):
        # print(f"{indent}Converting ndarray to list")
        obj = obj.tolist()
    # else:
    #     print(f"{indent}Value Type: {type(obj)}")
    return obj


def run(args):
    set_random_seed(seed=0)
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    # 현재 시간을 초 단위까지 가져오기
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"./logs/{args.task}/{args.backend}_{args.n_lambda_value}_{args.task_start_index}_{args.n_select_sample}"
    file_name += f'_fixedB' if not(args.beam_adjustment) else ''

    
    if args.task_end_index == -1:
        args.task_end_index = task.__len__()

    file = f'./logs/{args.task}/{args.backend}_{args.n_lambda_value}_{current_time}.json'
    # file = file_name + f"_{current_time}.json"


    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
        
    args.chat_model = Llama(args.backend)
    args = set_configs(args)
    for i in tqdm(range(args.task_start_index, args.task_end_index)):
        start_time = datetime.datetime.now()
        try:
            ys, info = solve(args, task, i)

        except Exception as e:
            print(f"[ERROR] Failed to solve task index {i}: {e}")
            import traceback
            traceback.print_exc()

            # 최소 로그 저장
            logs.append({
                'idx': i,
                'error': str(e),
                'usage_so_far': args.chat_model.get_llama_usage(),
                'elapsed_time': (datetime.datetime.now() - start_time).total_seconds()
            })

            # 현재까지 로그 저장
            logs = inspect_and_convert(logs)
            with open(file, 'w') as f:
                json.dump(logs, f, indent=4)
            continue  # 다음 task로 넘어가기

        # ===== 정상 처리 시 =====
        infos = [task.test_output(i, y) for y in ys]

        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        accs = [info['r'] for info in infos]
        # cnt_avg += sum(accs) / len(accs)
        # cnt_any += any(accs)

        info.update({
            'idx': i,
            'ys': ys,
            'infos': infos,
            'usage_so_far': args.chat_model.get_llama_usage(),
            'elapsed_time': elapsed_time,
            # 'cnt_avg': cnt_avg,
            # 'cnt_any': cnt_any
        })
        logs.append(info)

        # 로그 저장
        logs = inspect_and_convert(logs)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)

        print(info)
        print(i, 'sum(accs)', sum(accs), 'elapsed_time', elapsed_time, '\n')

    # ===== 전체 통계 =====
    n = args.task_end_index - args.task_start_index
    # print("Final Average Accuracy:", cnt_avg / n)
    # print("Final Any-Success Rate:", cnt_any / n)
    print("Total Token Usage:", args.chat_model.get_llama_usage())

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['llama-2-7b', 'llama-2-13b', 'llama-2-70b','llama-3.1-8b','llama-3.1-70b'], default='llama-2-7b')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--max_tokens', type=int, default=128)
    args.add_argument('--task', type=str, required=True, choices=['gsm8k', 'aqua', 'commonsenseqa', 'strategyqa', 'object_counting', 'date_understanding'])
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=-1)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--batch_size', type=int, default=1, help='size of batch for inference')  
    args.add_argument('--use_batch', default=False, action='store_true')
    args.add_argument('--method_generate', type=str, choices=['beam_propose', 'beam_sample'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy', 'beam_search','base_beam'], default='beam_search')
    args.add_argument('--n_generate_sample', type=int, default=1, help='number of beam sampling') 
    args.add_argument('--mini_n_generate_sample', type=int, default=1, help='number of mini sampling for one generation')
    args.add_argument('--n_select_sample', type=int, default=1, help='number of initial beam width')
    args.add_argument('--n_lambda_value', type=float, default=0.4)
    args.add_argument('--beam_adjustment', type=lambda x : x.lower() == 'true', default=True)
    args.add_argument('--uncertainty', type=str, choices=['shannon', 'gini', 'tsallis'], default='shannon')
    args.add_argument('--q_value', type=float, default=0.5, help='q value for Tsallis entropy')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)