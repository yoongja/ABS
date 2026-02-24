import re
import os
import sympy
import math
import pandas as pd
from ours.tasks.base import Task, DATA_PATH
from ours.prompts.game24 import * 

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='random100.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.rank = list(pd.read_csv(path)['Rank'])
        self.value_cache = {}
        self.steps = 4
        # self.stops = ['.\n\n', ":\n\n", "\n\n", " \n\n"]
        self.stops = []

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='', k:int=4) -> str:
        temp = y.strip().split('\n')
        current_numbers = get_current_numbers(y if y else x)
        # breakpoint()
        if len(temp) == 3:
            return cot_prompt.format(input=x) + y + "Answer:"
        else:
            return cot_propose_prompt.format(input=current_numbers, k=k)
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y

        else:
            prompt = cot_propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '').strip()
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, final_equation=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=x, current=current_numbers,candidate=last_line)
   
    @staticmethod
    def value_outputs_unwrap(x, y, value_outputs: list) :
        invalid = False
        text_output = [o['message']['content'].lower() for o in value_outputs][0]
        if text_output is not None:
            if 'no' in text_output:
                invalid = True
        valid_logprob = None
        # print(f"Text: {text_output}")
        # breakpoint()
        for c in value_outputs[0]["logprobs"]["content"]:
            if 'yes' in c['token'].lower():
                valid_logprob = c['logprob']
                break
                # breakpoint()
            else:
                top_logprobs = c['top_logprobs']
                for o in top_logprobs:
                    token = o.get('token', '').lower()
                    logprob = o.get('logprob', None)
                    if token == 'yes' and logprob is not None:
                        valid_logprob = logprob
                        break  # 가장 가까운 for 루프 종료
                if valid_logprob is not None:
                    break  # 바깥 루프도 종료
        # 확률 계산
        if valid_logprob is not None:
            probability = math.exp(valid_logprob)
            print(f"Text: {text_output}")
            print(f"Valid logprob: {valid_logprob}")
            print(f"Converted Probability: {probability:.4f}")
        else:
            print("No 'valid' token found.")
        return probability, invalid