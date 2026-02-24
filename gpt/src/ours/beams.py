from dataclasses import dataclass, field
from typing import List, Optional
import copy

@dataclass
class OneStepGen:
    one_step: str = ""
    lm_prob: Optional[float] = None
    eval_prob: Optional[float] = None
    trust_score: Optional[float] = None
    norm_trust: Optional[float] = None

    def update_eval_prob(self, eval_prob: float):
        if self.eval_prob is not None:
            print(f"[Warning] eval_prob already set: {self.eval_prob} → (ignored)")
            return
        self.eval_prob = float(eval_prob)
        
    def update_trust_score(self, trust_score: float):
        if self.trust_score is not None:
            print(f"[Warning] trust_score already set: {self.trust_score} → (ignored)")
            return
        self.trust_score = float(trust_score)

    def update_norm_trust(self, norm_trust: float):
        if self.norm_trust is not None:
            print(f"[Warning] norm_trust already set: {self.norm_trust} → (ignored)")
            return
        self.norm_trust = float(norm_trust)

    def pack_cur_score(self):
        vals = {
            'lm_prob': self.lm_prob,
            'eval_prob': self.eval_prob,
            'trust_score': self.trust_score,
            'norm_trust': self.norm_trust
        }
        # Ensure no None values
        clean = {k: (v if v is not None else None) for k, v in vals.items()}
        return clean


@dataclass
class BeamHypo:
    is_finished: bool = False
    length: int = 0
    generations: List[OneStepGen] = field(default_factory=list)

    def update_length(self):
        self.length = len(self.generations)

    def get_cur_gen(self, step: int) -> Optional[OneStepGen]:
        if 0 <= step < self.length:
            return self.generations[step]
        print(f"[Warning] Step {step} out of range (0, {self.length-1})")
        return None

    def add_step(self, gen: OneStepGen):
        self.generations.append(gen)
        self.update_length()

    def clone(self) -> 'BeamHypo':
        return copy.deepcopy(self)

    def update_finished(self):
        if self.is_finished:
            print(f"[Warning] is_finished already True → (ignored)")
            return
        self.is_finished = True

    def pack_the_cum_state(self, step: int) -> str:
        try:
            parts = []
            for i, gen in enumerate(self.generations[:step+1]):
                parts.append(str(gen.one_step))
            return ''.join(parts)
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def pack_the_beam_info(self, step: int) -> dict:
        try:
            cum_state = self.pack_the_cum_state(step)
            score = {}
            if 0 <= step < self.length:
                score = self.generations[step].pack_cur_score()
            return {
                'is_finished': bool(self.is_finished),
                'cum_state': cum_state,
                **score
            }
        except Exception as e:
            print(f"Error: {e}")
            return {}

    def pack_the_final_beam_info(self) -> dict:
        try:
            final_step = max(0, self.length - 1)
            cum_state = self.pack_the_cum_state(final_step)
            score = self.generations[final_step].pack_cur_score() if final_step < self.length else {}
            return {
                'is_finished': bool(self.is_finished),
                'length': int(self.length),
                'cum_state': cum_state,
                **score
            }
        except Exception as e:
            print(f"Error: {e}")
            return {}