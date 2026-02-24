import re
import os
import json
from ours.prompts.arithmetic.aqua import *
from ours.tasks.base import Task, DATA_PATH
from ours.models import gpt
import math
class AQUATask(Task):
    def __init__(self, file='your_path'):
        """
        file: a JSONL file containing AQUA problems.
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'aqua', file)
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]
            
        self.value_cache = {}
        self.steps = 16 # step 2개로 제한
        # self.stops = None
        self.stops = ['.\n\n', ":\n\n", "\n\n", " \n\n"]

    def __len__(self):
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def get_question(self, idx) -> str:
        return self.data[idx]['question']
    
    def get_options(self, idx):
        return self.data[idx]['options']
    
    def get_gt(self, idx: int) -> str:
        gt_ans = self.data[idx]['correct']
        # options = self.data[idx][]
        return gt_ans

    def test_output(self, idx: int, output: str):
        try:
            # 1) Load ground truth
            gt_letter = self.data[idx]['correct'].strip().lower()        # e.g. 'b'
            options   = self.data[idx]['options']                       # e.g. ["A)8","B)9",...]

            # Build letter→numeric map from options
            # matches 'A)8', 'B) 12.5', with optional spaces/commas
            opt_map = {}
            for opt in options:
                m = re.match(r'\s*([A-Ea-e])\)?\s*[:,]?\s*([\d,]+(?:\.\d+)?)', opt)
                if m:
                    letter = m.group(1).lower()
                    # remove commas before converting
                    num = float(m.group(2).replace(',', ''))
                    opt_map[letter] = num

            # Ground-truth numeric (may be None if gt_letter not found)
            gt_num = opt_map.get(gt_letter, None)

            # 2) Trim off any trailing question reprints
            if 'Q:' in output:
                output = output.split('Q:', 1)[0]

            # 3) Attempt to extract the answer block
            ans = None
            # first try after '###'
            parts = output.strip().split('###')
            if len(parts) > 1:
                ans = parts[-1].strip()
            else:
                # fallback to last double-newline chunk
                chunks = output.strip().split('\n\n')
                if len(chunks) >= 2:
                    ans = chunks[-2].strip()

            if not ans:
                print(f"[test_output] Couldn't isolate answer part in:\n{output}")
                return {'r': 0}

            ans = ans.lower()

            # 4) Extract letter prediction
            m = re.search(r'\(?([a-e])\)?', ans, re.IGNORECASE)
            pred_letter = m.group(1).lower() if m else None

            # 5) Extract numeric prediction (first number-like token)
            num_pat = r'\d+(?:,\d+)*(?:\.\d+)?'
            nums = re.findall(num_pat, ans)
            pred_num = None
            if nums:
                try:
                    pred_num = float(nums[0].replace(',', ''))
                except:
                    pred_num = None

            # 6) Compare
            letter_ok = (pred_letter == gt_letter)
            number_ok = (gt_num is not None and pred_num == gt_num)

            if letter_ok or number_ok:
                return {'r': 1}
            else:
                print(f"[test_output] Mismatch: pred_letter={pred_letter}, pred_num={pred_num}; "
                    f"expected letter={gt_letter}, num={gt_num}")
                return {'r': 0}

        except Exception as e:
            print(f"[test_output] Exception: {e}")
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