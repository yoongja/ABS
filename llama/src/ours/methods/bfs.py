import itertools
import numpy as np
import re
import math
import traceback
from src.ours.models_llama import *
from src.ours.utils import get_logger, interpolate, stop_sign

logger = get_logger(__name__)

def get_values(task, x, ys, n_generate_sample, llama, evaluation_config):
    n_batch = n_generate_sample
    values = []
    invalids = []
    
    for batch in itertools.zip_longest(*[iter(ys)] * n_batch, fillvalue=None):
        # batch = [y1, y2, .., y_batch]
        value_prompts = []
        for y in batch:
            value_prompts.append(task.value_prompt_wrap(x, y))
        _inputs = llama.tokenizer(value_prompts, return_tensors="pt", padding=True)
        prompts, attn_masks = _inputs.input_ids, _inputs.attention_mask
        value_outputs = llama.generate_output(prompts, attn_masks, evaluation_config, is_eval=True)
        for output in value_outputs:
            probability, invalid = task.value_outputs_unwrap([output])
            values.append(probability)
            invalids.append(invalid)
    return values, invalids

def beam_get_samples(task, x, y, n_generate_sample, prompt_sample, llama, generation_config):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    batch_prompt = [prompt] * n_generate_sample
    
    _inputs = llama.tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=True)
    prompts, attn_masks = _inputs.input_ids, _inputs.attention_mask
    
    samples = []
    try:
        outputs = llama.generate_output(prompts, attn_masks, generation_config)
        for output in outputs:
            samples.append(output)

    except Exception as e:
        print(f"Error during Llama sampling: {e}")
        return None

    samples_with_probs, lines, probs = [], [], []

    for i, sample in enumerate(samples):
        try:
            tokens, token_logprobs = [], []
            logprob, cur_line, cur_token_cnt = [], [], 0
            line = ""
            if not isinstance(sample, dict) or 'logprobs' not in sample or 'content' not in sample['logprobs']:
                print(f"Error: Sample {i} has an invalid structure.")
                return None

            if len(sample['logprobs']['content']) < 1:
                print(f"Error: Sample {i} is empty after first attempt.")
                return None

            for t in sample['logprobs']['content']:
                if not isinstance(t, dict) or 'token' not in t or 'logprob' not in t:
                    print(f"Error: Invalid token structure in sample {i}.")
                    return None

                tokens.append(t['token'])
                token_logprobs.append(t['logprob'])

            for i, t, lp in zip(range(len(tokens)), tokens, token_logprobs):
                min_log_prob = -50.0
                lp = max(lp, min_log_prob)
                logprob.append(lp)
                cur_line.append(t)
                cur_token_cnt += 1
            # compute mean log-prob and assemble line
            probs.append(math.exp(sum(logprob) / cur_token_cnt))
            line = ''.join(cur_line)

            if not(line.endswith('\n')) and not(line.endswith('.')):
                line += '.\n\n'
            lines.append(line)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            return None

    if not probs or not lines:
        print("Error: No valid samples were generated.")
        return None

    try:
        concat_lines = [y + _ for _ in lines]
        samples_with_probs.append(concat_lines)
        samples_with_probs.append(probs)
    
        return samples_with_probs
    
    except Exception as e:
        print(f"Error during final processing: {e}")
        return None

def ours_beam_search(value_probabilities, logit_probabilities, lambda_value, beam_adjustment, n_select_sample):
    if len(logit_probabilities) == 0 :
        return None
    # compute combined trust score
    current_combined_scores = (logit_probabilities ** lambda_value) * (value_probabilities ** (1 - lambda_value))
    # sort by descending combined score; argsort with negative for descending order
    sorted_indices = np.argsort(-current_combined_scores)
    descend_order_combined_scores = current_combined_scores[sorted_indices]

    descend_order_combined_scores = np.exp(descend_order_combined_scores) / np.sum(np.exp(descend_order_combined_scores))

    if beam_adjustment:
        # normalize to get sampling probabilities
        sampling_probs = descend_order_combined_scores / np.sum(descend_order_combined_scores)
        print(f"Initial sampling probabilities: {sampling_probs}")

        # compute max entropy for normalization
        max_probs = np.array([1/len(sampling_probs)] * len(sampling_probs))
        max_entropy = -np.sum(max_probs * np.log(max_probs + 1e-9))

        # compute normalized entropy for dynamic beam width adjustment
        entropy = -np.sum(sampling_probs * np.log(sampling_probs + 1e-9))
        norm_entropy = entropy/max_entropy

        selected_ids = list(range(len(current_combined_scores)))
        selection_info = []
        cumulative_prob = 0.0
        threshold = norm_entropy

        cumul_pass = False
        for s, c in enumerate(descend_order_combined_scores):  # iterate in descending score order
            original_index = int(sorted_indices[s])
            cumulative_prob += c
            selected_ids.append(original_index)
            info = {
                "Index": int(original_index),
                "logit": float(logit_probabilities[original_index]),
                "value": float(value_probabilities[original_index]),
                "Score": float(c),
                "CumulativeProb": float(cumulative_prob),
                "Threshold": float(threshold),
                "Pass": bool(cumulative_prob >= threshold)
            }
            selection_info.append(info)

            # stop when cumulative probability mass exceeds the entropy threshold
            if cumulative_prob >= threshold:
                cumul_pass = True
                break
        if cumul_pass == True:
            for info in selection_info:
                info["Pass"] = bool(True)
        # if threshold > 0.9, use interpolation to limit beam size
        if threshold > 0.9:
            print(f"threshold > 0.9, beam_size before adjustment: {len(selection_info)}")
            score = np.clip(np.mean(current_combined_scores), 0, 1)
            beam_size = int(interpolate(score, [0, 1], [10, 2]))
            selection_info = selection_info[:beam_size]
            selected_ids = sorted_indices[:beam_size]
            print(f"beam_size after adjustment: {beam_size}")

        if len(selected_ids) == 0:
            max_index = np.argmax(current_combined_scores)
            selected_ids = [max_index]
            print(f"No valid states found, selecting the max score index: {max_index}")

        print(f"Valid states (B(S_t)): {selected_ids}")
        beam = max(1, len(selected_ids))
        print(f"After beam size: {beam}")
    else:
        beam = n_select_sample
        selected_ids = sorted_indices[:beam]
        
        selection_info = [
            {
                "Index": int(s),
                "Score": float(c),
                "CumulativeProb": None,
                "Threshold": None,
                "Pass": None
            }
            for s, c in enumerate(current_combined_scores)
        ]
        norm_entropy = None
    return selected_ids, norm_entropy, selection_info

def solve(args, task, idx, to_print=True):  
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    already_solves = []
    stop_bol = False
    step = 0
    while stop_bol != True and step < 20:
        step += 1
        if args.method_generate == 'beam_sample':
            if step == 1:
                samples_with_probs = [beam_get_samples(task,x, y, args.n_generate_sample + 1, prompt_sample=args.prompt_sample, llama=args.chat_model, generation_config=args.generation_config) for y in ys]

            samples_with_probs = [beam_get_samples(task,x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, llama=args.chat_model, generation_config=args.generation_config) for y in ys]
            if samples_with_probs is None or any(s is None for s in samples_with_probs): 
                continue
            new_ys = [sample[0] for sample in samples_with_probs]
            logit_probabilities = [sample[1] for sample in samples_with_probs]
            logit_probabilities = list(itertools.chain(*logit_probabilities))
        
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        if args.method_evaluate == 'value':
            values, invalids = get_values(task, x, new_ys, n_generate_sample=args.n_generate_sample, llama=args.chat_model, evaluation_config=args.evaluation_config)
            print(f"Values: {values}, len: {len(values)}")

        if args.method_select == 'beam_search':
            values = np.array(values)
            logit_probabilities = np.array(logit_probabilities)
            selected_ids, entropy, selection_info = ours_beam_search(
                values, logit_probabilities,
                lambda_value=args.n_lambda_value, beam_adjustment=args.beam_adjustment, n_select_sample=args.n_select_sample
            )

        select_new_ys = []
        for select_id in selected_ids:
            if stop_sign(new_ys[select_id]):
                already_solves.append(new_ys[select_id])
                if len(already_solves) == args.n_select_sample :
                    stop_bol = True
                    break
            else:
                if invalids[select_id] == "True":
                    invalid_prompt = f'Prior reasoning had an error: {new_ys[select_id]} Let\'s fix that. Think step by step again, avoiding the mistake above.'
                    select_new_ys.append(invalid_prompt)
                else:
                    select_new_ys.append(new_ys[select_id])
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append({
            'step': step,
            'x': x,
            'ys': ys,
            'new_ys': new_ys,
            'values': values,
            'select_new_ys': select_new_ys,
            'beam': len(selected_ids),
            'entropy': entropy,
            'selection_info': selection_info,
            'logit_probabilities': logit_probabilities,
            'value_probabilities': values
        })
        ys = select_new_ys
        
        if stop_bol == True:
            return already_solves, {'steps': infos}

    if to_print: 
        print(already_solves)
        
    return already_solves, {'steps': infos}