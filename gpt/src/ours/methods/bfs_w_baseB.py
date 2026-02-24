import itertools
import numpy as np
import re
import math
from functools import partial
from src.ours.models import gpt
from src.ours.beams import BeamHypo, OneStepGen
from dataclasses import asdict
from src.ours.tasks.game24_ours import Game24Task


def interpolate_to_beam(mean_score, mean_score_range, beam_range):
    try:
        a, b = mean_score_range
        c, d = beam_range
        ratio = (mean_score - a) / (b - a)
        inter_value = c + ratio * (d - c)
        return int(inter_value)
    except Exception as e:
        print(f"[ERROR] interpolate_to_beam failed: {e}")
        return beam_range[-1]  # fallback to smallest beam
def get_value(task, x, y):
    try:
        value_prompt = task.value_prompt_wrap(x, y)
        value_outputs = gpt(value_prompt, temperature=0.5, max_tokens=5, n=1, logprobs=True, top_logprobs=20, stop=None)
        value, invalid = task.value_outputs_unwrap(x, y, value_outputs)
        return value, invalid
    except Exception as e:
        print(f"[ERROR] get_value failed: {e}")
        return 1e-6

def update_values(task, step, x, new_candidates):
    # values = []
    invalids = []
    for candidate in new_candidates:  # each partial output
        try:
            # breakpoint()
            cumul_gen = candidate.pack_the_cum_state(step)
            # breakpoint()
            eval_prob, invalid = get_value(task, x, cumul_gen)
            if eval_prob is None or eval_prob == 0:
                raise ValueError("Evaluation probability is invalid.")
            candidate.generations[step].update_eval_prob(eval_prob)
            invalids.append(invalid)
        except Exception as e:
            print(f"Error during self-evaluation: {e}")
            candidate.generations[step].update_eval_prob(1e-6)
    return invalids

def update_finished(candidate, step, final_beams, task):
    try:
        ans_pattern = ["###"]
        if task == 'date_understanding': ans_pattern.append("final answer")
        else: ans_pattern.append("the answer is" )
        
        if isinstance(task, Game24Task) and step == 3:
            candidate.update_finished()
            return
        
        for ans in ans_pattern:
            if ans in candidate.generations[step].one_step.lower():
                candidate.update_finished()
                final_beams.append(candidate)
    except Exception as e:
        print(f"[WARNING] update_finished failed: {e}")

def beam_get_samples(task, step, x, candidate, n_generate_sample, max_tokens, prompt_sample, stop):
    is_24 = isinstance(task, Game24Task)
    cumul_gen = candidate.pack_the_cum_state(step)
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, cumul_gen)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, cumul_gen)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # breakpoint()
    raw_samples = []
    n = n_generate_sample
    while n > 0:
        try:
            # outputs => len == 1(한 번에 생성하는 수)인 리스트
            outputs = gpt(prompt, max_tokens=max_tokens, n=1, logprobs=True, top_logprobs=0, stop=stop)
            sample = outputs[0]
            # 샘플 구조 확인
            if not isinstance(sample, dict) or 'logprobs' not in sample or 'content' not in sample['logprobs']:
                # breakpoint()
                print(f"Error: Sample {len(raw_samples)} has an invalid structure.")
                continue

            # 샘플 내용 비어있는 경우 처리
            if len(sample['logprobs']['content']) < 1:
                print(f"Error: Sample {len(raw_samples)} is empty after first attempt.")
                continue  # 비어 있는 경우 None 반환

            # 토큰과 로그 확률 추출
            for t in sample['logprobs']['content']:
                if not isinstance(t, dict) or 'token' not in t or 'logprob' not in t:
                    print(f"Error: Invalid token structure in sample {len(raw_samples)}.")
                    continue  # 구조 문제가 있으면 None 반환
            
            raw_samples.append(sample)
            n -= 1
            
        except Exception as e:
            print(f"Error during GPT sampling: {e}")
            return []  # GPT 호출 실패 시 None 반환
    # breakpoint()
    all_candidates_for_one = []
    # 각 sample에 대해 logprob 값 구하기
    for i, sample in enumerate(raw_samples):
        try:
            tokens, token_logprobs = [], []
            logprob, cur_line, cur_token_cnt = [], [], 0
            line = ""

            for t in sample['logprobs']['content']:       
                tokens.append(t['token'])
                token_logprobs.append(t['logprob'])

            # 로그 확률 계산
            for i, t, lp in zip(range(len(tokens)), tokens, token_logprobs):
                # inf나 nan -> 계산에서 제외
                if not math.isfinite(lp):
                    continue
                logprob.append(lp)
                cur_line.append(t)
                cur_token_cnt += 1

            # 확률 계산
            cur_lm_prob = math.exp(sum(logprob) / cur_token_cnt)
            # probs.append(math.exp(sum(logprob)))
            
            # 해당 단계에서 새로 생성된 문장
            line = ''.join(cur_line).strip()
            
            if not is_24:
                # 만약에 특수문자로 끝나지 않고, .으로도 끝나지 않으면 -> '.\n\n)
                if not(line.endswith('.')) and line[-1].isalnum():
                    line += ".\n\n"
                else:
                    line += "\n\n"
            else:
                line += "\n"

            # 새로 생성된 빔에 대해 처리
            new_gen = OneStepGen(one_step=line, lm_prob=cur_lm_prob, eval_prob=None, trust_score=None)
            # breakpoint()
            if step != 0: 
                new_beam = candidate.clone()
                new_beam.add_step(new_gen)
            else:
                candidate.add_step(new_gen)
                new_beam = candidate
            all_candidates_for_one.append(new_beam)
            # all_probs.append()
            

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            return [] # 개별 샘플 처리 중 오류 발생 시 None 반환
    # breakpoint()
    return all_candidates_for_one

def ours_beam_search(
    task,
    step,
    new_ids,
    cur_candidates,
    lambda_value,
    n_select_sample,
    final_beams,
    beam_adjustment=True,
    to_print=False
):
    """
    Performs entropy-thresholded beam search step.
    Returns (kept_beam, active_kept_beam, step_infos).
    """
    # 1) Compute trust scores
    all_trust = []
    # breakpoint()
    for cand in cur_candidates:
        try:
            lm_p = max(float(cand.generations[step].lm_prob or 0), 1e-9)
            ev_p = max(float(cand.generations[step].eval_prob or 0), 1e-9)
            log_lm = math.log(lm_p)
            log_ev = math.log(ev_p)
            trust = math.exp(lambda_value * log_lm + (1 - lambda_value) * log_ev)
        except Exception as e:
            print(f"[WARNING] Trust score calc error at step {step}: {e}")
            trust = 1e-6
        try:
            cand.generations[step].update_trust_score(trust)
        except:
            pass
        all_trust.append(trust)

    trust_np = np.array(all_trust, dtype=np.float32)
    total = float(np.sum(trust_np)) or 1e-9
    norm_trust = trust_np / total
    for cand, nt in zip(cur_candidates, norm_trust):
        try:
            cand.generations[step].update_norm_trust(nt)
        except:
            pass

    # 2) Determine beam_width
    try:
        max_ent = float(np.log(len(new_ids) or 1))
        entropy = float(-np.sum(norm_trust * np.log(norm_trust + 1e-9)))
        mean_tr = float(np.mean(trust_np))
        sorted_idx = np.argsort(-norm_trust)

        if beam_adjustment:
            norm_ent = entropy / max_ent
            if norm_ent > 0.9:
                beam_width = min(interpolate_to_beam(mean_tr, [0,1], [10,2]), int(len(cur_candidates)))
                
            else:
                cumsum = np.cumsum(norm_trust[sorted_idx])
                idx = next((i for i, v in enumerate(cumsum) if v >= norm_ent), len(cumsum))
                beam_width = max(1, idx - 1)
            if beam_width > 16: beam_width = 16
        else:
            norm_ent = -1
            beam_width = int(n_select_sample)
       
        beam_width = int(beam_width)
        kept_idx = sorted_idx[:beam_width]
        kept_beam = [cur_candidates[i] for i in kept_idx]
    except Exception as e:
        print(f"[WARNING] Beam width selection failed: {e}")
        beam_width = int(1)
        kept_idx = sorted_idx[:beam_width]
        kept_beam = [cur_candidates[i] for i in kept_idx]
        entropy = 0.0
        mean_tr = 0.0
        norm_ent = 0.0
        sorted_idx = list(range(len(cur_candidates)))

    # 3) Check finished and collect active beams
    active_kept_beam = []
    for kb in kept_beam:
        try:
            update_finished(kb, step, final_beams, task)
        except:
            pass
        if not kb.is_finished:
            active_kept_beam.append(kb)

    # 4) Logging
    if to_print:
        if to_print:
            print(
                f"[Beam Step {step}] "
                f"candidates={int(len(cur_candidates))}, "
                f"beam_width={beam_width}, "
                f"entropy={entropy:.6f}, "
                f"norm_entropy={norm_ent:.6f}, "
                f"mean_trust={mean_tr:.6f}"
            )

    # 5) Pack step_infos
    cand_beams = []
    for i in sorted_idx.tolist():
        try:
            info = cur_candidates[i].pack_the_beam_info(step)
        except Exception as e:
            print(f"[WARNING] pack_beam_info failed at idx {i}: {e}")
            info = {}
        cand_beams.append({
            "index": int(i),
            "is_selected": (int(i) in kept_idx),
            **info
        })

    step_infos = {
        "beam_width": beam_width,
        "num_candidates": int(len(cur_candidates)),
        "entropy": entropy,
        "norm_entropy": norm_ent,
        "mean_trust": mean_tr,
        "selected_beam_idx": kept_idx,
        "candidate_info": cand_beams
    }

    return kept_beam, active_kept_beam, step_infos


# idx -> task input idx
# 하나의 input에 대해 수행됨
def solve(args, task, idx, to_print=True):
    max_step = 20 if args.task != 'game24' else 3
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature, max_tokens=args.max_tokens)
    print(gpt)
    final_beams = []
    final_beams_info = []
    
    x = task.get_input(idx)  # input
    all_step_infos = []  # 모든 중간 단계의 로깅 데이터를 포함하는 리스트  
    kept_beams = [BeamHypo() for i in range(args.n_generate_sample)]     # 초기 샘플링 수만큼 빔 객체 생성  
    step = 0

    while len(final_beams) < args.n_select_sample:
        # 1) Generation
        new_candidates = [] # list of BeamHypo
        # generation
        if args.method_generate == 'beam_sample':
            try:
                sample_num = 1 if step == 0 else args.n_generate_sample
                for c in kept_beams:
                    samples = beam_get_samples(
                        task, step, x, c,
                        sample_num,
                        max_tokens=args.max_tokens,
                        prompt_sample=args.prompt_sample,
                        stop=task.stops
                    )
                    if samples:
                        new_candidates.extend(samples)
  
            except Exception as e:
                print(f"[ERROR] Generation failed at step {step}: {e}")
                new_candidates = []
                continue
            
        new_ids = list(range(len(new_candidates)))
        
        # 2) Evaluation
        if args.method_select != 'base_beam':
            if args.method_evaluate == 'value':
                invalids = update_values(task, step, x, new_candidates)

        # selection
       
        if (args.method_select == 'beam_search') or (args.method_select == 'base_beam') :
            new_kept_beams, active_kept_beams, step_infos = ours_beam_search(args.task, step, new_ids, new_candidates,lambda_value=args.n_lambda_value,beam_adjustment=args.beam_adjustment, n_select_sample=args.n_select_sample, final_beams=final_beams,to_print=to_print)
            step_info = {"step": step}
            step_info.update(step_infos)
            
            if to_print:
                    bw = step_infos.get('beam_width', '-')
                    ent = step_infos.get('norm_entropy', '-')
                    print(f"[STEP {step}] Beam width={bw}, Entropy={ent}")

            # 만약 모든 선택된 상태가 is_finished이고, final beams 의 개수가 모자란 경우 
            # -> 현재 final beams 를 그냥 최종 결과로
            
            if (len(active_kept_beams) == 0) and (len(final_beams) != args.n_select_sample):
                all_step_infos.append(step_info)
                break
            # breakpoint()
            if isinstance(task, Game24Task) and step == max_step:
                final_beams = new_kept_beams[: args.n_select_sample]
                all_step_infos.append(step_info)
                break
            
            if step > max_step:
                # 1) If final_beams already has some beams, keep them as-is.
                if final_beams:
                    all_step_infos.append(step_info)
                    break
                else:
                    # 2) Otherwise, take the top n_select_sample from new_kept_beam
                    #    (assuming new_kept_beam is already sorted by trust score descending)
                    final_beams = new_kept_beams[: args.n_select_sample]
                    all_step_infos.append(step_info)
                    break
            
            # if a selected node is evaluated as invalid, add feedback to the current node
            # for i in step_infos['selected_beam_idx']:
            #     if invalids[i]:
            #         cur_step = new_candidates[i].get_cur_gen(step).one_step
            #         invalid_prompt = f'Prior reasoning had an error: {cur_step} Let\'s fix that. Think step by step again, avoiding the mistake above.'
            #         new_candidates[i].generations[step].one_step = invalid_prompt

            
            # 해당 단계에서의 선택된 빔 리스트 업데이트
            kept_beams = active_kept_beams
      
     
        # 로그 데이터에 Beam 크기와 Entropy 추가
        all_step_infos.append(step_info)
        # increase step count
        step += 1
        
     
    # if final_beams_info already exists and you want to add to it:
    # After beam search completes, trim to top n_select_sample by last-step trust score
    if len(final_beams) > args.n_select_sample:
        # Sort beams by the trust_score of their last generation in descending order
        final_beams = sorted(
            final_beams,
            key=lambda beam: (
                beam.generations[-1].trust_score
                if beam.generations and beam.generations[-1].trust_score is not None
                else float('-inf')
            ),
            reverse=True
        )[: args.n_select_sample]
            
    final_beams_info.extend(
        fb.pack_the_final_beam_info() for fb in final_beams
    )
    
    return final_beams_info, {'steps': all_step_infos}, final_beams

