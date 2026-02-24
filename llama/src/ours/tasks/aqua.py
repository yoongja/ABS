import re
import os
import json
import math
from src.ours.prompts.arithmetic.aqua import *
from src.ours.tasks.base import Task, DATA_PATH

class AQUATask(Task):
    def __init__(self, file='your_path'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'aqua', file)
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]
            
        self.value_cache = {}
        self.steps = 12
        self.stops = ['.\n\n', ":\n\n", "\n\n", " \n\n"]

    def __len__(self):
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        try:
            gt = self.data[idx]['correct'].strip().lower()
            options = self.data[idx]['options']

            gt_full = None
            for option in options:
                if option.strip().lower().startswith(gt):
                    gt_full = option.strip().lower()
                    break

            valid_gts = [gt, gt_full]

            if 'Q:' in output:
                parsed_output = output.strip().split('Q:')[0]
            else:
                parsed_output = output

            try:
                ans = parsed_output.strip().split('###')[-1].strip().lower()
            except IndexError:
                ans = None

            if not ans or ans == "":
                try:
                    ans = parsed_output.strip().split('\n\n')[-2].strip().lower()
                except IndexError:
                    ans = None

            match = re.search(r'\(([a-e])\)', ans)
            if match:
                pred = re.sub(r'[().]', '', match.group(1))
            else:
                print(f"There is no answer in final output: {output}")
                return {'r': 0}

            if pred in valid_gts:
                return {'r': 1}
            else:
                print(f"Prediction '{pred}' does not match valid gts: {valid_gts}")
                return {'r': 0}

        except Exception as e:
            print(f"Error during test_output: {e}")
            return {'r': 0}
        
    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        question = x['question']
        ops = [('(' + o).lower() for o in x['options']]
        options = ' '.join(ops)
        return cot_prompt.format(question=question, options=options) + y

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        question = x['question']
        ops = [('(' + o).lower() for o in x['options']]
        options = ' '.join(ops)
        return value_prompt.format(question=question, options=options, output=y)
    
    @staticmethod
    def value_outputs_unwrap(value_outputs: list) -> float:
        invalid = False
        text_output = [o['text'].lower() for o in value_outputs][0]
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