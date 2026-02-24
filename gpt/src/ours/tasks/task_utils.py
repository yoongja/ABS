import os
import json
import jsonlines

from ours.tasks.base import DATA_PATH

tasks = ['aqua', 'gsm8k', 'commonsenseqa', 'strategyqa', 'date_understanding', 'game24']

def get_input_path(task):
    if task in tasks:
        category = ""
        file_name = ""
        if task == 'aqua' or task == 'gsm8k':
            category, file_name = "Arithmetic", "test_origin.jsonl"
        elif task == 'commonsenseqa':
            category, file_name = "Commonsense", "dev_rand_split.jsonl"
        elif task == 'strategyqa':
            category, file_name = "Commonsense", "dev.json"
        elif task == 'date':
            category, file_name = "Symbolic", "task.json"
        elif task == 'game24':
            return os.path.join(DATA_PATH, "24", "24.csv")
            
        return os.path.join(DATA_PATH, category, task, file_name)
    else:
        raise NotImplementedError
    
    
def get_question_and_options(task, data):
    question = ""
    options = ""
    
    if task == 'commonsenseqa':
        ops = []
        question = data['question']['stem']
        for o in data['question']['choices']:
            ops.append('(' + o["label"].lower() + ')' + ' ' + o['text'])
        options = '\n'.join(ops) + '\n'
    elif task  == 'strategyqa':
        question = data['question']
    elif task == 'gsm8k':
        question = data['question']
    elif task == 'aqua':
        question = data['question']
        options = "\n".join(data['options'])
    elif task == 'data_understanding':
        question = data['input']
    else:
        print("##### Input question errror: Can't get question and options from input file")
        return None, None
    
    return question, options
        
def get_prompt_path(task):
    module_path = "tot.prompts."
    if task == 'gsm8k' or task == 'aqua':
        module_path += f"arithmetic.{task}"
    elif task == 'commonsenseqa' or task == 'strategyqa':
        module_path += f"commonsense.{task}"
    elif task == 'date_understanding':
        module_path += f"symbolic.{task}"
    elif task == 'game24':
        module_path += f"{task}"
    return module_path