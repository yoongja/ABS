import re
import os
import sympy
import pandas as pd
import json
import importlib
from ours.tasks.base import Task
from ours.tasks.task_utils import *

def import_prompt_module(name):
    module_path = get_prompt_path(name)
    try:
        module = importlib.import_module(module_path)
        return module
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        return None
    
class GeneralTask(Task):
    def __init__(self, name):
        """"
        """
        super().__init__()
        path = get_input_path(name)
        if name == 'game24':
            self.data = list(pd.read_csv(path)['Puzzles'])
        else: 
            try:
                if os.path.exists(path):
                    with open(path, mode='r') as f:
                        self.data = [json.loads(line) for line in f]
            except OSError as e:
                print(f"Error in input file path '{path}': {e}")
        
        self.task = name
        self.steps = 4  # max step for beam search
        self.stops = ['.\n\n']  # stop sign for sampling generation
        self.value_cache = {}
        
        
    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def get_task_name(self) -> str:
        return self.task

    def test_output(self, idx: int, output: str):
        return {'r': 1}

    @staticmethod
    def cot_prompt_wrap(self, x: dict, y:str='') -> str:
        name = self.get_task_name()
        question, options = get_question_and_options(name,x)
        module = import_prompt_module(name)
        if question is not None and options is not None:
            if len(options) != 0:
                return module.cot_prompt.format(input=question, options=options) + y
            else:
                return module.cot_prompt.format(input=question) + y
        else:
            return module.cot_prompt.format(input=x) + y
              
    @staticmethod
    def value_prompt_wrap(self, x: dict, y: str) -> str:
        name = self.get_task_name()
        question, options = get_question_and_options(name,x)
        module = import_prompt_module(name)
        if question is not None and options is not None:
            if len(options) != 0:
                return module.value_prompt.format(input=question, options=options) + y
            else:
                return module.value_prompt.format(input=question) + y
        else:
            return module.value_prompt.format(input=x) + y
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        text_output = [o['text'].lower() for o in value_outputs]
        # Define mapping for value names
        value_map = {'invalid': 0.001, 'valid': 20} 
        
        value_list = []
        for t in text_output:
            if t.count('invalid') > 0:
                value_list.append(value_map['invalid'])
            elif t.count('valid') > 0:
                value_list.append(value_map['valid'])
            else:
                value_list.append(0) 
        
        # Calculate the final value based on the counts of each value name
        value = sum(value_list)
        return value