import re
import os
import jsonlines
from ours.tasks.base import Task, DATA_PATH
from ours.prompts.commonsense.commonsenseqa import *
import math 

class CommonsenseqaTask(Task):
    """
    Input (x)   : a text question
    Options (o) : a text options (5)
    Output  (y) : a sequence of texts for the final answer
    Reward (r)  : 

    Input Example: What do people use to absorb extra ink from a fountain pen?
    
    Option Example:
    (a) shirt pocket
    (b) calligrapher’s hand
    (c) inkwell
    (d) desk drawer
    (e) blotter
    
    Output Example:
    The answer must be an item that can absorb ink.
    Of the above choices, only blotters are used to absorb ink.
    So the answer is (e).
    
    """
    
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
        self.stops = ['.\n\n', ' \n\n', '\n\n', ":\n\n"]
        self.value_cache = {}
        
    def __len__(self) -> int:
        return len(self.data)

    # x : 하나의 데이터셋 dict 데이터 -> input: question 반환
    def get_input(self, idx: int) -> str:
        raw = self.data[idx]
        input = {"question": raw['question']['stem'], "answerKey": raw['answerKey'], "choices": raw['question']['choices']}
        return input
    
    def get_question(self, idx: int) -> str:
        return  self.data[idx]['question']['stem']
    
    def get_options(self, idx: int) :
        ops = []
        choices = self.data[idx]['question']['choices']
        for o in choices:
            ops.append('(' + o["label"] + ')' + ' ' + o['text'])
        return ops

    def get_gt(self, idx:int) -> str:
        return self.data[idx]['answerKey']
        
    def test_output(self, idx: int, output: str):
        input = self.get_input(idx)
        
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

    # get_samples 에서 사용 -> 해당 프롬프트통해 다음 단계 도출
    @staticmethod
    def cot_prompt_wrap(x: dict, y:str='') -> str:
        question = x['question']
        ops = []
        for o in x['choices']:
            ops.append('(' + o["label"].lower() + ')' + ' ' + o['text'])
        options = ' '.join(ops) 
        return cot_prompt.format(question=question, options=options) + y

    # get_value에서 사용
    # y: 현재 생성된 후보 ys 중 한개
    @staticmethod
    def value_prompt_wrap(x: dict, y: str) -> str:
        # last_line = y.strip().split('\n')[-1]
        question = x['question']
        ops = []
        for o in x['choices']:
            ops.append('(' + o["label"].lower() + ')' + ' ' + o['text'])
        options = ' '.join(ops)
        
        return value_prompt.format(question=question, options=options, output=y)
    
    # get_value 에서 사용
    # gpt에서 생성된 value_outputs parsing -> 해당 y에 대한 value 도출
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        """
        Returns probability based solely on the first 'valid' token's logprob.
        """
        probability = 0.0
        try:
            # 모든 토큰의 top_logprobs를 순회
            contents = value_outputs[0].get("logprobs", {}).get("content", [])
            for token_info in contents:
                top_logprobs = token_info.get("top_logprobs", [])
                for entry in top_logprobs:
                    token = entry.get('token', '').lower()
                    logprob = entry.get('logprob')
                    # 'valid' 포함 & 'invalid' 미포함일 때 첫 번째만 사용
                    if logprob is not None and 'valid' in token and 'invalid' not in token:
                        # 첫 valid 토큰 로그확률로 확률 계산
                        probability = math.exp(logprob + 1e-9)
                        # 바로 반환 (첫 번째만 고려)
                        print(f"First valid token: {token}, logprob: {logprob}")
                        print(f"Converted Probability: {probability:.6f}")
                        return probability
        except Exception as e:
            print(f"[ERROR] value_outputs_unwrap failed: {e}")

        # valid 토큰이 없거나 예외 시 0 반환
        print("No valid token found; returning probability=0.0")
        return probability