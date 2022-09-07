import numpy as np

from action_value_table import ActionValueTable

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3  # agents actions
GAMMA = 0.99
STEP_SIZE = 0.25
EPSILON = 0.1


class QLearningAgent():
    '''
    Implement your code for a Q-learning agent here. We have provided code implementing the action-value
    table in action_value_table.py. Here, you will implement the get_action(), get_greedy_action() and update() methods.
    '''

    def __init__(self, dimension):
        self.actions = [UP, DOWN, LEFT, RIGHT]
        num_actions = len(self.actions)
        self.values = ActionValueTable(dimension, num_actions)
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.epsilon = EPSILON


    def update(self, state, action, reward, next_state, done):
        '''
        This function will update the values stored in self.value using Q-learning. 

        HINT: Use self.values.get_value and self.values.set_value
        HINT: Remember to add a special case to handle the terminal state

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze
            
            action : (int) the action taken at state

            reward : float

        returns:
            action : (int) a epsilon-greedy action for state 
        '''

        ### YOUR CODE HERE ###
        old = self.values.get_value(state, action)
        max_action_next = self.get_action(next_state)
        max_qsa_next = self.values.get_value(next_state,max_action_next) 
        new = old + self.step_size*(reward + self.gamma*(max_qsa_next - old))
        self.values.set_value(state, action, new)
        return self.get_action(next_state) 

        raise NotImplementedError


    def get_action(self, state):
        '''
        This function returns an action from self.actions given a state. 

        Implement this function using an epsilon-greedy policy. 

        HINT: use np.random.rand() to generate random numbers
        HINT: If more than one action has maximum value, treat them all as the greedy action. In other words,
        if there are b greedy actions, each should have probability epsilon/b + |A|, where |A| is 
        the number of actions in this state.

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze

        returns:
            action : (int) a epsilon-greedy action for state 

        '''

        ### YOUR CODE HERE ###
        if np.random.rand()< self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_list=[self.values.get_value(state, self.actions[0]), self.values.get_value(state, self.actions[1]), self.values.get_value(state, self.actions[2]), self.values.get_value(state, self.actions[3])]
            q_list = np.array(q_list)
            ii =q_list.max()
            max = 0
            for i in range(4):
                if q_list[i] == ii:
                    max = i

            return max        

    def get_greedy_action(self, state):
        '''
        This function returns an action from self.actions given a state. 

        Implement this function using a greedy policy, i.e. return the action with the highest value
        HINT: If more than more than one action has maximum value, uniformly randomize amongst them

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze

        returns:
            action : (int) a greedy action for state 
        '''

        ### YOUR CODE HERE ###

        q_list=[self.values.get_value(state, self.actions[0]), self.values.get_value(state, self.actions[1]), self.values.get_value(state, self.actions[2]), self.values.get_value(state, self.actions[3])]
        q_list = np.array(q_list)
        ii =q_list.max()
        max = 0
        for i in range(4):
            if q_list[i] == ii:
                max = i

        return max