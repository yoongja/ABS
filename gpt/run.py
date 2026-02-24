import os
import json
import argparse
import datetime
import random
from tqdm import tqdm

from src.ours.tasks import get_task
from src.ours.methods.bfs_w_stop import solve
from src.ours.models import gpt_usage

import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def inspect_and_convert(obj, depth=0):
    """Recursively inspect types and convert ndarray to list."""
    indent = "  " * depth  # Indentation for better readability
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = inspect_and_convert(value, depth + 1)
    elif isinstance(obj, list):
        obj = [inspect_and_convert(item, depth + 1) for item in obj]
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()
 
    return obj

def run(args):
    set_random_seed(seed=0)
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")

    file_name = f"./logs/{args.task}/{args.temperature}_ld{args.n_lambda_value}_s{args.n_generate_sample}_b{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_"
    file_name += f'_fixedB' if not(args.beam_adjustment) else ''
    file_name += f'_notIP' if not(args.do_interpolate) else ''

    
    if args.naive_run:
        file = f'./logs/{args.task}/{args.temperature}_lambda_{args.n_lambda_value}_epsilon_{args.n_epsilon_value}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_{current_time}.json'
    else:
        file = file_name + f'_{current_time}.json'

    os.makedirs(os.path.dirname(file), exist_ok=True)

    if args.task_end_index == -1:
        args.task_end_index = task.__len__()
        
    is_24 = args.task == 'game24'
        
    for i in tqdm(range(args.task_start_index, args.task_end_index)):
        start_time = datetime.datetime.now() 

        # solve
        ys, info, f_beams = solve(args, task, i, to_print=True)

        # log
        idx = i if not(is_24) else task.rank[i]
        cur_input = task.get_question(i) if not is_24 else task.get_input(i)
        cur_option = task.get_options(i) if not is_24 else None
        gt = task.get_gt(i) if not is_24 else None
        cum_states = [i.pack_the_cum_state(i.length-1) for i in f_beams]
        infos = [task.test_output(i, y) for y in cum_states]  # correctness for each final beam (r: 0 or 1)
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        accs = [info['r'] for info in infos]
        if sum(accs) == 0: 
            cnt_avg += 0
        else:    
            cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)

        cur_log = {
            'idx': idx,
            'input': cur_input,
            **({'option': cur_option, 'gt': gt} if not is_24 else {}),
            'steps': info,
            'final_beams': ys,
            'infos': infos,
            'usage_so_far': gpt_usage(args.backend),
            'elapsed_time': elapsed_time,
            'cnt_avg': cnt_avg,
            'cnt_any': cnt_any
        }
        cur_log = inspect_and_convert(cur_log)
        logs.append(cur_log)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)        
        
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, 'elapsed_time', elapsed_time, '\n')

    # Final statistics
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-4o-mini'], default='gpt-4o-mini')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--max_tokens', type=int, default=128)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'gsm8k', 'aqua', 'commonsenseqa', 'strategyqa', 'date_understanding'])
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=-1)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['beam_propose', 'beam_sample'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy', 'beam_search', 'base_beam'], default='beam_search')
    args.add_argument('--n_generate_sample', type=int, default=4)  # sampling number for generation
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=2) # beam width for classic beam search
    args.add_argument('--n_lambda_value', type=float, default=0.4)  # hyperparameter for trust score
    args.add_argument('--do_interpolate', type=lambda x: x.lower() == 'true', default=True)
    args.add_argument('--beam_adjustment', type=lambda x: x.lower() == 'true', default=True)
   

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)