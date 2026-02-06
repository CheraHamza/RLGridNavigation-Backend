import random

ACTIONS = ['up', 'down', 'left', 'right']

class RandomAgent:
    def act(self, state):
        return random.choice(ACTIONS)