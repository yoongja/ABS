import re
import os
import json
import pandas as pd
import math
from ours.tasks.base import Task, DATA_PATH
from ours.prompts.commonsense.strategyqa import * 

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
        self.stops = ['.\n\n', ' \n\n', '\n\n', ":\n\n"]
        self.value_cache = {}
        
    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        raw = self.data[idx]
        input = {'question': raw['question'], 'answer': raw['answer']}
        return input
    
    def get_question(self, idx: int) -> str:
        return self.data[idx]['question']
    def get_options(self, idx):
        return []
    def get_gt(self, idx: int): 
        return self.data[idx]['answer']
    
    def test_output(self, idx: int, output: str):
        # breakpoint()
        input = self.get_input(idx)
        gt = input['answer']
        ans = ''
        
        # 이상한 output parsing
        if 'Q:' in output:
            parsed_output = output.split('Q:')[0].strip()
        else: parsed_output = output.strip()
        
        # 최종 결과에서 정답 파싱
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

    # get_samples 에서 사용 -> 해당 프롬프트통해 다음 단계 도출
    @staticmethod
    def cot_prompt_wrap(x: dict, y:str='') -> str:
        question = x['question']
        return cot_prompt.format(question=question) + y

    @staticmethod
    def value_prompt_wrap(x: dict, y: str) -> str:
        # last_line = y.strip().split('\n')[-1]
        question = x['question']
        return value_prompt.format(question=question, output=y)

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