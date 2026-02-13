import random
import pickle

ACTIONS = ['up', 'down', 'left', 'right']

class RandomAgent:
    def act(self, state):
        return random.choice(ACTIONS)
    
class QLearningAgent:
    def __init__(self):
        # Hyperparameters
        self.alpha = 0.1       # learning rate
        self.gamma = 0.99      # Discount factor
        self.epsilon = 1.0     # Exploration rate (start high)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Q-Table
        self.q_table = {}

        # Memory for the update step
        self.prev_state = None
        self.prev_action = None

    def get_q(self, state, action):
        """Helper to get Q-value, defaulting to 0.0 if unknown."""
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in ACTIONS}
        return self.q_table[state_key][action]
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        state_key = tuple(state)

        # Initialize state in Q-table if new
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in ACTIONS}

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        
        # Exploitation (argmax)

        q_values = self.q_table[state_key]
        max_q = max(q_values.values())
        # Handle ties randomly to prevent getting stuck
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, current_state, reward, done):
        """
        Updates Q(prev_state, prev_action) using the reward
        and the max Q value of the current_state.
        """
        if self.prev_state is None or self.prev_action is None:
            return # Can't learn on the very first step of a run
        
        # 1. Retrieve Q(s, a)
        old_value = self.get_q(self.prev_state, self.prev_action)

        # 2. Calculate Max Q(s', a')
        next_max = 0.0
        if not done:
            next_state_key = tuple(current_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {a: 0.0 for a in ACTIONS}
            next_max = max(self.q_table[next_state_key].values())

        # 3. Compute Target
        target = reward + (self.gamma * next_max)

        # 4. Update Q-value
        new_value = old_value + self.alpha * (target - old_value)
        self.q_table[tuple(self.prev_state)][self.prev_action] = new_value

        # 5. Handle Episode End (Decay Epsilon)
        if done:
            self.prev_state = None
            self.prev_action = None
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_memory(self, state, action):
        """Remember what we just did for the NEXT update."""
        self.prev_state = state
        self.prev_action = action
        
    def to_bytes(self):
        """Serializes the agent state to bytes (for DB storage)."""
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon
        }
        return pickle.dumps(data)

    def from_bytes(self, byte_data):
        """Loads agent state from bytes (from DB)."""
        data = pickle.loads(byte_data)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]