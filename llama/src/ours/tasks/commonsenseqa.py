import re
import os
import jsonlines
import math
from src.ours.tasks.base import Task, DATA_PATH
from src.ours.prompts.commonsense.commonsenseqa import * 

class CommonsenseqaTask(Task):
    def __init__(self, file='your_path'):
        super().__init__()
        
        # read input file
        path = os.path.join(DATA_PATH, 'Commonsense', file)
        try:
            if os.path.exists(path):
                with jsonlines.open(path, mode='r') as reader:
                    self.data = [row for row in reader]
        except OSError as e:
            print(f"Error in input file path '{path}': {e}")
            
        self.steps = 4  # task reasoning steps
        self.stops = ['.\n\n', '. ', '\n\n']
        self.value_cache = {}
        
    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        raw = self.data[idx]
        input = {"question": raw['question']['stem'], "answerKey": raw['answerKey'], "choices": raw['question']['choices']}
        return input

    def test_output(self, idx: int, output: str):
        input = self.get_input(idx)
        
        # trim off any trailing question reprints
        if 'Q:' in output:
            parsed_output = output.split('Q:')[0].strip()
        else: parsed_output = output.strip()
        
        ans = parsed_output.split('###')[-1].lower().strip()
        match = re.search(r'\(([a-e])\)', ans)
        if match:
            ans_op = re.sub(r'[().]', '', match.group(1))
        else:
            print(f"There is no answer in final output: {output}")
            return {'r': 0}
        
        gt = input['answerKey'].lower()
        if gt in ans_op:
            return {'r': 1}
        else:
            return {'r': 0}

    @staticmethod
    def cot_prompt_wrap(x: dict, y:str='') -> str:
        question = x['question']
        ops = []
        for o in x['choices']:
            ops.append('(' + o["label"].lower() + ')' + ' ' + o['text'])
        options = ' '.join(ops) 
        return cot_prompt.format(question=question, options=options) + y

    @staticmethod
    def value_prompt_wrap(x: dict, y: str) -> str:
        temp_y = y.split("\n")
        sen_only = []
        for s in temp_y:
            if len(s) != 0 and ("Fact" not in s) and ("Reasoning" not in s):
                sen_only.append(s)
        new_y = "\n\n".join(sen_only)        
        question = x['question']
        return value_prompt.format(question=question, output=new_y)
    
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