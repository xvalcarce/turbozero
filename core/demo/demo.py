
import random
import torch
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.algorithms.load import init_evaluator
from core.env import Env
from IPython.display import clear_output
import os

class Demo:
    def __init__(self,
        evaluator: Evaluator,
        evaluator_args: dict,
        manual_step: bool = False
    ):
        
        self.manual_step = manual_step
        self.evaluator = evaluator
        assert self.evaluator.env.parallel_envs == 1
        self.evaluator_args = evaluator_args
        self.evaluator.reset()

    def run(self, print_evaluation: bool = False, print_state: bool = True, interactive: bool =True):
        self.evaluator.reset()
        while True:
            if print_state:
                print(self.evaluator.env)
            if self.manual_step:
                input('Press any key to continue...')
            _, _, value, _, terminated = self.evaluator.step(**self.evaluator_args)
            if interactive:
                clear_output(wait=True)
            else:
                os.system('clear')
            if print_evaluation and value is not None:
                print(f'Evaluation: {value[0]}')
            if terminated:
                print('Game over!')
                print('Final state:')
                print(self.evaluator.env)
                break

class TwoPlayerDemo(Demo):
    def __init__(self,
        evaluator,
        evaluator_args,
        evaluator2,
        evaluator2_args,
        manual_step: bool = False
    ) -> None:
        super().__init__(evaluator, evaluator_args, manual_step)
        self.evaluator2 = evaluator2
        assert self.evaluator2.env.parallel_envs == 1
        self.evaluator2_args = evaluator2_args
        self.evaluator2.reset()
    
    def run(self, print_evaluation: bool = False, print_state: bool = True, interactive: bool =True):
        self.evaluator.reset()
        self.evaluator2.reset()
        p1_turn = random.choice([True, False])
        p1_started = p1_turn
        actions = None
        while True:
            

            active_evaluator = self.evaluator if p1_turn else self.evaluator2
            other_evaluator = self.evaluator2 if p1_turn else self.evaluator
            evaluator_args = self.evaluator_args if p1_turn else self.evaluator2_args
            if print_state:
                active_evaluator.env.print_state(int(actions.item()) if actions is not None else None)
            if self.manual_step:
                input('Press any key to continue...')
            _, _, value, actions, terminated = active_evaluator.step(**evaluator_args)
            other_evaluator.step_evaluator(actions, terminated)
            if interactive:
                clear_output(wait=True)
            else:
                os.system('clear')
            if print_evaluation and value is not None:
                print(f'Evaluation: {value[0]}')
            if terminated:
                print('Game over!')
                print('Final state:')
                active_evaluator.env.print_state(int(actions.item()))
                self.print_rewards(p1_started)
                break
                
            p1_turn = not p1_turn

    def print_rewards(self, p1_started: bool):
        reward = self.evaluator.env.get_rewards(torch.tensor([0 if p1_started else 1]))[0]
        if reward == 1:
            print(f'Player 1 ({self.evaluator.__class__.__name__}) won!')
        elif reward == 0:
            print(f'Player 2 ({self.evaluator2.__class__.__name__}) won!')
        else:
            print('Draw!')