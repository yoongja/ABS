import sys
import time
import re
import os
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    GenerationConfig,
    StoppingCriteriaList
)
from .utils import get_logger

logger = get_logger(__name__)

def clean_generated_text(text):
    # subword marker 제거
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    return text.strip()

def set_configs(args):
    tokenizer = args.chat_model.tokenizer
    # beam state generation -> sampling
    args.generation_config = GenerationConfig(
        max_new_tokens = args.max_tokens,
        # num_return_sequences=args.mini_n_generate_sample,
        temperature = args.temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,   
        eos_token_id=tokenizer.eos_token_id
    )
    
    # self-evaluation -> greedy
    args.evaluation_config = GenerationConfig(
        max_new_tokens=5,
        min_new_tokens=1,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,   
        eos_token_id=tokenizer.eos_token_id
    )
    
    return args

class NewlinesStoppingCriteria(StoppingCriteria):
    '''
    Stop when encountering any of stop_strs
    '''
    def __init__(self, tokenizer, stop_strs: List[str], max_check_tokens=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strs = stop_strs
        self.max_check_tokens = max_check_tokens
        
        self.prompt_lengths = None  # Prompt length for each batch sequence
        self.stopping_mask = None   # Whether each batch sequence is interrupted
        self.stop_positions = None  # Position where each batch sequence stopped       
        
    def reset_stopping_config(self, batch_size, prompt_lengths: List[int]):
        if len(prompt_lengths) != batch_size:
            logger.warning(f"Warning: prompt_lengths ({len(prompt_lengths)}) does not match the batch size ({batch_size})")
            
        self.prompt_lengths = prompt_lengths
        self.stopping_mask = [False] * batch_size
        self.stop_positions = [None] * batch_size
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        device = input_ids.device
        batch_ret, seq_len = input_ids.shape    # batch_ret = batch_size * num_return_sequences
        # breakpoint()
        
        if (self.prompt_lengths) is None :
            logger.warning("Warning: prompt_lengths is not set. Set all prompt_lengths to 0.")
            self.prompt_lengths = [0] * batch_ret
            
        # Prepare a tensor of False flags
        stop_flags = torch.zeros(batch_ret, dtype=torch.bool, device=device)
        
        for i in range(batch_ret):
            if self.stopping_mask[i]:  
                stop_flags[i] = True  # Returns True if the sample has already stopped
                continue
            
            prompt_len = self.prompt_lengths[i]
            cur_len = seq_len
            
            # if not enough new tokens generated yet, skip newline check
            if cur_len < (prompt_len + self.max_check_tokens):  
                continue
                
            # check last tokens for newline stop patterns    
            recent_ids = input_ids[i, prompt_len:].tolist()[-self.max_check_tokens:]
            recent_txt = self.tokenizer.decode(recent_ids, skip_special_tokens=True)
            if any(recent_txt.endswith(s) for s in self.stop_strs):
                self.stopping_mask[i] = True  
                self.stop_positions[i] = cur_len - 1
                stop_flags[i] = True

        return stop_flags    # Return a boolean tensor: True for sequences that should stop


class Llama:
    def __init__(self, model_name=None):
        
        def _load_model_and_tokenizer(model_path):
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                padding_side='left',    # for K-V cache efficiency
                cache_dir = '/data3/hg_weight/hg_weight'
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                device_map='auto',
                cache_dir = '/data3/hg_weight/hg_weight'
            )
            # tokenizer.add_special_tokens({"pad_token": "[PAD]"})    # assign new pad token            
            # model.resize_token_embeddings(len(tokenizer))
            # tokenizer.model_max_length = model.config.max_position_embeddings
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        
        # =============================================== initialization ===============================================
        if model_name == 'llama-2-7b':
            self.model_path = 'meta-llama/Llama-2-7b-hf'
            # self.model_path = '/data3/hg_weight/hg_weight/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'
        elif model_name == 'llama-2-13b':
            self.model_path = 'meta-llama/Llama-2-13b-hf'
            # self.model_path = '/data3/hg_weight/hg_weight/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1'
        elif model_name == 'llama-2-70b':
            self.model_path = 'meta-llama/Llama-2-70b-hf'
            # self.model_path = '/data3/hg_weight/hg_weight/models--meta-llama--Llama-2-70b-hf/snapshots/3aba440b59558f995867ba6e1f58f21d0336b5bb'
        elif model_name == 'llama-3.1-8b':
            self.model_path = 'meta-llama/Llama-3.1-8B'
        elif model_name == 'llama-3.1-70b':
            self.model_path = 'meta-llama/Llama-3.1-70B'
        self.model, self.tokenizer = _load_model_and_tokenizer(self.model_path)
        
        self.completion_tokens, self.prompt_tokens = 0, 0
        stop_strs = [":\n\n", "\n\n", " \n\n", ".\n\n"]
        self.newline_criteria = NewlinesStoppingCriteria(tokenizer=self.tokenizer, stop_strs=stop_strs)
        
    
    def get_llama_usage(self):
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens}
    
    # Count usage of tokens
    def update_llama_usage(self, prompt, completion):
        self.completion_tokens += completion
        self.prompt_tokens += prompt
        
    def _process_log_probs_and_tokens_with_topk(
            self,
            logits: torch.FloatTensor,
            gen_ids: torch.LongTensor, 
            pad_token_id: int, 
            top_k: int = 1000
        ) -> List[Dict[str, Any]]:
            """
            Given per-step logits and generated token IDs, return a list of dicts:
            - token: string of the generated token
            - logprob: log probability of that token
            - top_logprobs: list of top_k {token, logprob} for that step, excluding pad tokens
            """
            logits = logits.masked_fill(logits == float('-inf'), -1e9)  # replace any -inf logits (padding) with large negative value for stability
            log_probs = F.log_softmax(logits.float(), dim=-1)  # (gen_len, vocab_size)

            result = []
            
            # iterate over each generation time step
            for idx, token_id in enumerate(gen_ids.tolist()):
                if token_id == pad_token_id:
                    continue

                token_logprob = log_probs[idx, token_id].item()  # selected token logprob

                # top-k ids and their logprobs
                topk_logprobs, topk_ids = log_probs[idx].topk(top_k)
                top_entries = []
                for tok_id, lp in zip(topk_ids.tolist(), topk_logprobs.tolist()):
                    if tok_id == pad_token_id:
                        continue
                    tok = self.tokenizer.convert_ids_to_tokens(tok_id)
                    tok = tok.replace('▁', ' ').replace('<0x0A>', '\n')
                    top_entries.append({"token": tok, "logprob": lp})

                sel_tok = self.tokenizer.convert_ids_to_tokens(token_id)
                sel_tok = sel_tok.replace('▁', ' ').replace('<0x0A>', '\n')
                
                result.append({
                    "token": sel_tok,
                    "logprob": token_logprob,
                    "top_logprobs": top_entries
                })
                
            return result      

    @torch.inference_mode()
    def generate_output(
        self, 
        input_ids: torch.LongTensor, 
        attn_masks: torch.LongTensor, 
        generation_config, 
        is_eval=False
    ) -> List[Dict[str, Any]]:
        """
        Generate text with logprob extraction and unified sequence/score handling.

        Returns:
            A list of dictionaries, each containing:
            - "text": generated string
            - "logprobs": {"content": [token-level logprob info]}
        """
        batch_size = input_ids.shape[0]
        device = self.model.device
        num_ret = generation_config.num_return_sequences
        
        # breakpoint()
        if not(is_eval):
            prompt_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist() # Calculate the length of all batch sequence prompts
        else:
            prompt_lengths = [input_ids.shape[1]] * batch_size
            
        expanded_prompt_lengths = []
        for pl in prompt_lengths:
            expanded_prompt_lengths.extend([pl] * num_ret)
        
        # Resets the stop criteria when generating states
        stoppers = None
        if not(is_eval):
            self.newline_criteria.reset_stopping_config(batch_size*num_ret, expanded_prompt_lengths)
            stoppers = StoppingCriteriaList([self.newline_criteria])

        outputs = self.model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attn_masks.to(device),
            generation_config=generation_config,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=stoppers,
        )   
        # breakpoint()
        sequences = outputs.sequences                          # (batch*ret, total_seq_len)
        batch_ret, seq_len = sequences.size()
        sequences = sequences.view(batch_size, num_ret, seq_len)    # (batch, ret, seq_len)
        
        # scores: List[(batch*ret, vocab_size)] 길이=gen_len
        stacked = torch.stack(outputs.scores, dim=0).to(device)                 # (gen_len, batch*ret, vocab_size)
        gen_len, _, vocab_size = stacked.shape
        scores = stacked.view(gen_len, batch_size, num_ret, vocab_size)
        scores = scores.permute(1, 2, 0, 3)                    # (batch, ret, gen_len, vocab)

            
        if not(is_eval):
            stop_positions = self.newline_criteria.stop_positions
            # sequences: (batch, ret, seq_len) -> mask pad after stop
            for i in range(batch_size):
                for j in range(num_ret):
                    stop_idx = stop_positions[i]
                    # stop_idx points to the last valid generated token; mask tokens after it
                    if (
                        stop_idx is not None 
                        and stop_idx >= expanded_prompt_lengths[i]
                        and stop_idx < seq_len - 1
                    ):
                        sequences[i, j, stop_idx+1:] = self.tokenizer.pad_token_id  # pad from stop_idx+1 onward
                        gen_stop = stop_idx - expanded_prompt_lengths[i] + 1
                        # if 0 <= gen_stop < scores.size(2):
                        #     scores[i, j, gen_stop+1:, :] = float('-inf')    # mask from the next generated token
        
        result = []
        total_completion = 0
        
        for i in range(batch_size):
            # if is_eval:
            #     breakpoint()
            
            all_gen_ids = [sequences[i, j][expanded_prompt_lengths[i]:].tolist() for j in range(num_ret)]
            # decoded_texts = self.tokenizer.batch_decode(all_gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_texts = self.tokenizer.batch_decode(
                all_gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_texts = [t.strip() for t in decoded_texts]
            for j in range(num_ret):
                gen_ids = all_gen_ids[j]
                logits = scores[i, j]
                max_steps = logits.size(0)  # generated sequence length
                trimmed_ids = gen_ids[:max_steps]
                log_ids = torch.tensor(trimmed_ids, device=device)
                top_k=1000 if is_eval else 5
                logprob_info = self._process_log_probs_and_tokens_with_topk(
                    logits=logits, gen_ids=log_ids, pad_token_id=self.tokenizer.pad_token_id, top_k=top_k
                )

                # 텍스트 변환
                text = decoded_texts[j]
                text = clean_generated_text(decoded_texts[j])
                result.append({
                    "text": text,
                    "logprobs": {"content": logprob_info}
                })
                total_completion += len(trimmed_ids)

        
        total_prompt = sum(expanded_prompt_lengths)
        self.update_llama_usage(total_prompt, total_completion)
        
        return result
