
class ReplayLogger:

    def __init__(self):

        raise NotImplementedError

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def log(self, state, action, reward, next_state, done):
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get(self):
            
        return self.states, self.actions, self.rewards, self.next_states, self.dones
    
    def clear(self):
            
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []