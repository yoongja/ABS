import re
import os
import json
import math
import pandas as pd
from src.ours.tasks.base import Task, DATA_PATH
from src.ours.prompts.commonsense.strategyqa import * 

class StrategyqaTask(Task):
    """
    Input (x)   : a text question
    Output (y)  : a text answer

    Q: Could Brooke Shields succeed at University of Pennsylvania?
    A:
    Brooke Shields graduated from Princeton University.
    According to US news, Princeton University and University of Pennsylvania are ranked as the number 1 and 6 national college, respectively.
    This can indicate that Princeton University is about as academically rigorous as the University of Pennsylvania.
    Thus, Brooke Shields could also succeed at University of Pennsylvania.
    So the answer is yes.
    """
    
    def __init__(self, file='your_path'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'Commonsense', file)
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    self.data = json.load(file)
        except OSError as e:
            print(f"Error in input file path '{path}': {e}")
            
        self.steps = 4
        self.stops = ['.\n\n', '\n', '. ']
        self.value_cache = {}
        
        
    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        raw = self.data[idx]
        input = {'question': raw['question'], 'answer': raw['answer']}
        return input
    
    def test_output(self, idx: int, output: str):
        input = self.get_input(idx)
        gt = input['answer']
        ans = ''

        # trim off any trailing question reprints
        if 'Q:' in output:
            parsed_output = output.split('Q:')[0].strip()
        else: parsed_output = output.strip()

        # extract final answer from output
        ans = parsed_output.split('###')[-1].lower().strip().replace('.', '')
        ans = ans.split(" ")
        if len(ans) > 0:
            pred = [i for i in ans if i in ("yes", "no")]
            
        if len(ans) == 0 or len(pred) == 0:
            print(f"There is no answer in final output: {output}")
            return {'r': 0}
        
        pred = pred[0]
        
        if "yes" in pred: 
            ans_tf = True
        else:
            ans_tf = False

        if gt == ans_tf:
            return {'r': 1}
        else:
            return {'r': 0}
        
    @staticmethod
    def cot_prompt_wrap(x: dict, y:str='') -> str:
        question = x['question']
        return cot_prompt_v5.format(question=question) + y

    @staticmethod
    def value_prompt_wrap(x: dict, y: str) -> str:      
        question = x['question']
        return value_prompt.format(question=question, output=y)
    
    @staticmethod
    def value_outputs_unwrap(value_outputs: list) -> float:
        invalid = False
        text_output = [o['text'].lower() for o in value_outputs][0]
        if text_output is not None:
            if 'no' in text_output:
                invalid = True
        valid_logprob = None
        for c in value_outputs[0]["logprobs"]["content"]:
            if (any(x in c['token'].lower() for x in ['yes', 'correct'])) and ('incorrect' not in c['token'].lower()):
                valid_logprob = c['logprob']
                break
            else:
                top_logprobs = c['top_logprobs']
                for o in top_logprobs:
                    token = o.get('token', '').lower()
                    logprob = o.get('logprob', None)
                    if (token == 'yes' or token == "correct") and logprob is not None:
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