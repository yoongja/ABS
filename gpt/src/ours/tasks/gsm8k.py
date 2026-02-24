import re
import os
import sympy
import pandas as pd
import json
from ours.prompts.arithmetic.gsm8k import *
from ours.tasks.base import Task, DATA_PATH
from ours.models import gpt
import math

class GSM8KTask(Task):
    def get_current_numbers(y: str) -> str:
        reasoning_part = y.split("**Answer:**")[0].strip()        
        return reasoning_part

    def __init__(self, file='your_path'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'gsm8k', file)
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]
            
        self.value_cache = {}
        self.steps = 5
        self.stops = ['.\n\n', ":\n\n", "\n\n", " \n\n"]

    def __len__(self):
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]['question']
    def get_question(self, idx):
        return self.data[idx]['question']
    def get_options(self, idx):
        return []
    def get_gt(self, idx):
        raw_ans = self.data[idx]['answer'].split('\n')[-1]
        cleaned = raw_ans.replace('#', '').replace(',', '').strip()
        digits_only = re.findall(r'\d+', cleaned)
        if digits_only:
            return int(digits_only[0])
        else:
            print(f"[ERROR] No valid integer found in answer string: '{raw_ans}' (idx={idx})")
            return None

    def test_output(self, idx: int, output: str):
        try:
            gt_ans = self.get_gt(idx)
            if 'Q:' in output:
                parsed_output = output.strip().split('Q:')[0]
            else: parsed_output = output
            
            ans = parsed_output.strip().split('###')[-1].strip().lower()
            pattern = r"\d+(?:,\d+)*(?:\.\d+)?"
            ans_list = re.findall(pattern, ans)
            
            if len(ans_list) == 0:
                print(f"There is no answer in final output: {output}")
                return {'r': 0}
            else:
                pred = int(float(ans_list[0].replace(',', '')))

            if pred == gt_ans:
                return {'r': 1}
            else:
                return {'r': 0}

        except Exception as e:
            print(f"Error during test_output: {e}")
            return {'r': 0}

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(question=x) + y

    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return value_prompt.format(question=x, output=y)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        """
        Returns probability based solely on the first 'valid' token's logprob.
        """
        probability = 0.0
        try:
            contents = value_outputs[0].get("logprobs", {}).get("content", [])
            for token_info in contents:
                top_logprobs = token_info.get("top_logprobs", [])
                for entry in top_logprobs:
                    token = entry.get('token', '').lower()
                    logprob = entry.get('logprob')
                    # use first token containing 'valid' but not 'invalid'
                    if logprob is not None and 'valid' in token and 'invalid' not in token:
                        probability = math.exp(logprob + 1e-9)
                        print(f"First valid token: {token}, logprob: {logprob}")
                        print(f"Converted Probability: {probability:.6f}")
                        return probability
        except Exception as e:
            print(f"[ERROR] value_outputs_unwrap failed: {e}")

        print("No valid token found; returning probability=0.0")
        return probability