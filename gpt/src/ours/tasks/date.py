import re
import os
import json
import pandas as pd
import math
from ours.tasks.base import Task, DATA_PATH
from ours.prompts.symbolic.date_understanding import * 

class DateUnderstandingTask(Task):
    """
    Input (x)   : a text question
    Output (y)  : a text answer

    Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
    A:
    Let's think step by step.
    Yesterday was 04/30/2021.
    One day after 04/30/2021 is 05/01/2021.
    Final Answer: 05/01/2021
    
    data['examples] = {'input': "","target_scores": {
        "05/01/2021": 1,
        "02/23/2021": 0,
        "03/11/2021": 0,
        "05/09/2021": 0,
        "06/12/2021": 0,
        "04/29/2021": 0
      }}
    """
    
    def __init__(self, file='your_path'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'Symbolic', file)
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    self.data = json.load(file)
                    self.data = self.data['examples']
        except OSError as e:
            print(f"Error in input file path '{path}': {e}")
            
        self.steps = 7
        self.stops = ['.\n\n', '\n\n', " \n\n", ":\n\n"]
        self.value_cache = {}
        
        
    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def get_question(self, idx):
        return self.data[idx]['input']
    
    def get_options(self, idx):
        return []
    def get_gt(self,idx):
        gts = self.data[idx]['target_scores']
        gt = [key for key, value in gts.items() if value == 1][0]
        return gt
    
    def test_output(self, idx: int, output: str):
        input = self.get_input(idx)
        gts = input['target_scores']  # target dict: keys are date strings, value 1 = correct
        gt = [key for key, value in gts.items() if value == 1][0]
        
        if 'Q:' in output:
            parsed_output = output.strip().split('Q:')[0]
        else: parsed_output = output
        
        ans = ''
        ans = parsed_output.strip().split('###')[-1].strip().lower()
        
        if 'answer:' in ans:
            ans = parsed_output.split(':')[-1].replace('.', '').strip()
        else:
            ans = parsed_output.split(' ')[-1].replace('.', '').strip()
        
        if len(ans) > 0:
            is_right = gt == ans
        else: return {'r' : 0}    
        
        if is_right: return {'r': 1}
        else: return {'r': 0}

    @staticmethod
    def cot_prompt_wrap(x: dict, y:str='') -> str:
        question = x['input']
        return cot_prompt.format(question=question) + y

    @staticmethod
    def value_prompt_wrap(x: dict, y: str) -> str:
        question = x['input']
        return value_prompt.format(question=question, output=y)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        invalid = False
        text_output = [o['message']['content'].lower() for o in value_outputs][0]
        if text_output is not None:
            if 'no' in text_output:
                invalid = True
        valid_logprob = None
        print(f"Text: {text_output}")
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
    