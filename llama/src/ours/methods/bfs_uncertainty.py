import itertools
import numpy as np
import re
import math
import traceback
from src.ours.models_llama import *
from src.ours.utils import get_logger, interpolate, stop_sign

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

    # 각 sample에 대해 logprob 값 구하기
    for i, sample in enumerate(samples):
        try:
            tokens, token_logprobs = [], []
            logprob, cur_line, cur_token_cnt = [], [], 0
            line = ""
            # 샘플 구조 확인
            if not isinstance(sample, dict) or 'logprobs' not in sample or 'content' not in sample['logprobs']:
                print(f"Error: Sample {i} has an invalid structure.")
                return None  # 구조 문제가 있으면 즉시 None 반환

            # 샘플 내용 비어있는 경우 처리
            if len(sample['logprobs']['content']) < 1:
                print(f"Error: Sample {i} is empty after first attempt.")
                return None  # 비어 있는 경우 None 반환

            # 토큰과 로그 확률 추출
            for t in sample['logprobs']['content']:
                if not isinstance(t, dict) or 'token' not in t or 'logprob' not in t:
                    print(f"Error: Invalid token structure in sample {i}.")
                    return None  # 구조 문제가 있으면 None 반환
                
                tokens.append(t['token'])
                token_logprobs.append(t['logprob'])
            
            # 로그 확률 계산
            for i, t, lp in zip(range(len(tokens)), tokens, token_logprobs):
                min_log_prob = -50.0
                lp = max(lp, min_log_prob)
                logprob.append(lp)
                cur_line.append(t)
                cur_token_cnt += 1
            # 확률 계산 및 문장 연결
            probs.append(math.exp(sum(logprob) / cur_token_cnt))
            # probs.append(math.exp(sum(logprob)))
            line = ''.join(cur_line)
        
            if not(line.endswith('\n')) and not(line.endswith('.')):
                line += '.\n\n'
            lines.append(line)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            return None  # 개별 샘플 처리 중 오류 발생 시 None 반환

    if not probs or not lines:  # 모든 샘플이 비었을 경우
        print("Error: No valid samples were generated.")
        return None

    # 결과 병합
    try:
        concat_lines = [y + _ for _ in lines]
        # total = sum(probs)
        # normal_logprobs = [x / total for x in probs]
        samples_with_probs.append(concat_lines)
        samples_with_probs.append(probs)
    
        return samples_with_probs
    
    except Exception as e:
        print(f"Error during final processing: {e}")
        return None  # 결과 병합 중 오류 발생 시 None 반환

def et_beam_search(value_probabilities, logit_probabilities, lambda_value, beam_adjustment, n_select_sample, q_value, uncertain_type='shannon'):
    print(f"========================beam adjustment: {beam_adjustment}==========================")
    if len(logit_probabilities) == 0 :
        return None
    # 결합 점수 계산
    current_combined_scores = (logit_probabilities ** lambda_value) * (value_probabilities ** (1 - lambda_value))
    current_combined_scores = np.log(current_combined_scores)

    # 원래의 Idx를 찾기위함
    sorted_indices = np.argsort(-current_combined_scores)  # 음수 부호로 내림차순 정렬
    descend_order_combined_scores = current_combined_scores[sorted_indices]

    descend_order_combined_scores = np.exp(descend_order_combined_scores) / np.sum(np.exp(descend_order_combined_scores))

    if beam_adjustment:
        norm_value = 0
        # 정규화
        sampling_probs = descend_order_combined_scores / np.sum(descend_order_combined_scores)
        print(f"Initial sampling probabilities: {sampling_probs}")
        
        # calculate normalized shannon entropy (default)
        if uncertain_type == "shannon":
            # entropy normalize를 위한 max entropy 계산
            max_probs = np.array([1/len(sampling_probs)] * len(sampling_probs))
            max_entropy = -np.sum(max_probs * np.log(max_probs + 1e-9))

            # 동적 빔 너비 조정을 위한 엔트로피 계산
            entropy = -np.sum(sampling_probs * np.log(sampling_probs + 1e-9))
            norm_entropy = entropy/max_entropy           

            norm_value = norm_entropy
        
        # calculate normaized gini index
        elif uncertain_type == "gini":
            k = len(descend_order_combined_scores)
            gini_index = 1.0 - (np.sum(np.square(descend_order_combined_scores)))
            gini_max = 1.0 - (1.0/k)
            norm_gini = gini_index / gini_max
            
            norm_value = norm_gini 
            
        elif uncertain_type == "tsallis":
            q = q_value
            k = len(descend_order_combined_scores)
            uniform = np.ones(k) / k
            max_tsallis = (1.0 - np.sum(uniform ** q)) / (q - 1)
            tsallis = (1.0 - np.sum(descend_order_combined_scores ** q)) / (q - 1)
            norm_tsallis = tsallis / max_tsallis
            
            norm_value = norm_tsallis
        
        threshold = norm_value    
        selected_ids = list(range(len(current_combined_scores)))
        selection_info = []  # 결과를 저장할 리스트    
        cumulative_prob = 0.0
        cumul_pass = False
        
        for s, c in enumerate(descend_order_combined_scores):  # 높은 순으로 점수 정렬되어 있음
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
            # 멈출 조건: 누적 확률 질량이 엔트로피 기준을 초과
            if cumulative_prob >= threshold:
                cumul_pass = True
                break
        if cumul_pass == True:
            for info in selection_info:
                info["Pass"] = bool(True)
        # 0.9에 맞춰서
        if threshold > 0.9:
            print(f"threshold 0.9이상 beam_size : {len(selection_info)}")
            # 원래의 평균을 고려함
            score = np.clip(np.mean(current_combined_scores), 0, 1)
            beam_size = int(interpolate(score, [0, 1], [10, 2]))
            selection_info = selection_info[:beam_size]
            selected_ids = sorted_indices[:beam_size]
            print(f"조정 이후 beam_size : {beam_size}")

        if len(selected_ids) == 0:
            max_index = np.argmax(current_combined_scores)
            selected_ids = [max_index]
            print(f"No valid states found, selecting the max score index: {max_index}")

        print(f"Valid states (B(S_t)): {selected_ids}")
        beam = max(1, len(selected_ids))
        print(f"After beam size: {beam}")
        
    else:
        selected_ids = list(range(len(current_combined_scores)))
        norm_value = None
        selection_info = []
        beam = n_select_sample
        print(f"After beam size: {beam}")
        for i, c in enumerate(descend_order_combined_scores):  # 높은 순으로 점수 정렬되어 있음
            while i < beam:
                original_index = int(sorted_indices[i])
                is_pass = True if i < beam else False
           
                selected_ids.append(original_index)
                info = {
                        "Index": int(i),
                        "logit": float(logit_probabilities[original_index]),
                        "value": float(value_probabilities[original_index]),
                        "Score": float(c),
                        "Pass": bool(is_pass)
                    }
                selection_info.append(info)
        print(f"Valid states (B(S_t)): {selected_ids}")
    # breakpoint()
    return selected_ids, norm_value, selection_info

# idx -> task input idx
def solve(args, task, idx, to_print=True):  
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    already_solves = []
    stop_bol = False
    step = 0
    # for step in range(task.steps):
    while stop_bol != True and step < 20:
        # generation
        step += 1
        if args.method_generate == 'beam_sample':
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
            selected_ids, entropy, selection_info = et_beam_search(
                values, 
                logit_probabilities,
                lambda_value=args.n_lambda_value, 
                beam_adjustment=args.beam_adjustment, 
                n_select_sample=args.n_select_sample, 
                q_value=args.q_value,
                uncertain_type=args.uncertainty
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

        info = {
            'step': step,
            'x': x,
            'ys': ys,
            'new_ys': new_ys,
            'values': values,
            'select_new_ys': select_new_ys,
            'beam': len(selected_ids),  # 현재 Beam 크기
            'entropy': entropy,  # 현재 Entropy 값
            'selection_info': selection_info,  # 선택 정보
            'logit_probabilities': logit_probabilities,  # 현재 로짓 확률
            'value_probabilities': values  # 현재 가치 확률
        }
        if args.uncertainty != "shannon":
            info['Uncertainty'] = args.uncertainty
        
        infos.append(info)
        ys = select_new_ys
        if stop_bol == True:
            return already_solves, {'steps': infos}

    if to_print: 
        print(already_solves)
        
    return already_solves, {'steps': infos}