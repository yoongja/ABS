import re
import os
import sympy
import pandas as pd
import json
import math
from src.ours.prompts.arithmetic.gsm8k import *
from src.ours.tasks.base import Task, DATA_PATH

B_INST, E_INTS = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class GSM8KTask(Task):
    def get_current_numbers(y: str) -> str:
        # Split the output at 'Answer:' to isolate the reasoning part
        reasoning_part = y.split("**Answer:**")[0].strip()        
        return reasoning_part

    def __init__(self, file='your_path'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'gsm8k', file)
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]
            
        self.value_cache = {}
        self.steps = 7
        self.stops = ['.\n\n']

    def __len__(self):
        return len(self.data)

    def get_input(self, idx: int) -> str:
        cur = self.data[idx]
        question = cur['question']
        ans = cur['answer'].split("####")[1].strip()
        cur_input = {'question': question, 'answer': ans}
        return cur_input

    def test_output(self, idx: int, output: str):
        try:
            gt_ans = int(self.data[idx]['answer'].split("####")[1].strip())
            if 'Q:' in output:
                parsed_output = output.strip().split('Q:')[0]
            else: parsed_output = output
            
            ans = parsed_output.strip().split('###')[-1].strip().lower()
            
            ans_list = re.findall(r'\d+\.?\d*', ans)
            
            if len(ans_list) == 0:
                print(f"There is no answer in final output: {output}")
                return {'r': 0}
            else:
                pred = int(float(ans_list[0]))

            if pred == gt_ans:
                return {'r': 1}
            else:
                return {'r': 0}

        except Exception as e:
            print(f"Error during test_output: {e}")
            return {'r': 0}

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(question=x['question']) + y

    
    @staticmethod
    def value_prompt_wrap(x: dict, y: str) -> str:
        return value_prompt.format(question=x['question'], output=y)
    
    @staticmethod
    def value_outputs_unwrap(value_outputs: list) -> float:
        invalid = False
        text_output = [o['text'].lower() for o in value_outputs][0]
        if text_output is not None:
            if 'no' in text_output:
                invalid = True
        valid_logprob = None
        for c in value_outputs[0]["logprobs"]["content"]:
            if 'yes' in c['token'].lower():
                valid_logprob = c['logprob']
                break
            else:
                top_logprobs = c['top_logprobs']
                for o in top_logprobs:
                    token = o.get('token', '').lower()
                    logprob = o.get('logprob', None)
                    if token == 'yes' and logprob is not None:
                        valid_logprob = logprob
                        break  # break inner loop
                if valid_logprob is not None:
                    break  # break outer loop
        if valid_logprob is not None:
            probability = math.exp(valid_logprob)
            print(f"Text: {text_output}")
            print(f"Valid logprob: {valid_logprob}")
            print(f"Converted Probability: {probability:.4f}")
        else:
            print("No 'valid' token found.")
        return probability, invalid