import numpy as np
import gym
import math

class Enviroment:
    def doAction(self, action):
        pass

    def GetCurrentState(self):
        pass

    def getPossibleActions(self):
        pass

    def render(self):
        pass

    def reset(self):
        pass

class LunarLanderEnviroment(Enviroment):
    def __init__(self, renderMode, discrete, seed = None) -> None:
        super().__init__()
        self.discrete = discrete
        self.env = gym.make('LunarLander-v2', render_mode=renderMode)
        self.currentState = self.ConvertState(self.env.reset(seed=np.random.randint(0, 100) if seed == None else seed)[0])

    def ConvertState(self, state):
        f1 = round(state[0], 1) if self.discrete else state[0]
        f2 = round(state[1], 1) if self.discrete else state[1]
        f3 = round(state[2], 1) if self.discrete else state[2]
        f4 = round(state[3], 1) if self.discrete else state[3]
        f5 = round(state[0], 1) if self.discrete else state[4]
        f6 = round(state[1], 1) if self.discrete else state[5]
        f7 = round(state[2], 1) if self.discrete else state[6]
        f8 = round(state[3], 1) if self.discrete else state[7]
        return (f1, f2, f3, f4, f5, f6, f7, f8)

    def doAction(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.currentState = self.ConvertState(observation)
        #print(terminated)
        if action == 2:
            reward += 0.3
        if action == 3 or action == 1:
            reward += 0.03
        return reward, terminated, truncated, info
    
    def GetCurrentState(self):
        return self.currentState
    
    def getPossibleActions(self):
        return [0, 1, 2, 3]

    def reset(self):
        self.currentState = self.ConvertState(self.env.reset()[0])

    def render(self):
        self.env.render()


class CartPoleEnviroment(Enviroment):
    def __init__(self, renderMode, discrete, seed = None) -> None:
        super().__init__()
        self.discrete = discrete
        self.env = gym.make('CartPole-v1', render_mode=renderMode)
        self.currentState = self.ConvertState(self.env.reset(seed=np.random.randint(0, 100) if seed == None else seed)[0])

    def ConvertState(self, state):
        f1 = round(state[0], 1) if self.discrete else state[0]
        f2 = round(state[1], 1) if self.discrete else state[1]
        f3 = round(state[2], 1) if self.discrete else state[2]
        f4 = round(state[3], 1) if self.discrete else state[3]
        return (f1, f2, f3, f4)

    def doAction(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.currentState = self.ConvertState(observation)
        #print(terminated)
        return reward, terminated, truncated, info
    
    def GetCurrentState(self):
        return self.currentState
    
    def getPossibleActions(self):
        return [0, 1]

    def reset(self):
        self.currentState = self.ConvertState(self.env.reset()[0])

    def render(self):
        self.env.render()

class WalkingCliffEnviroment(Enviroment):
    def __init__(self, renderMode, seed = None) -> None:
        super().__init__()
        self.env = gym.make('CliffWalking-v0', render_mode=renderMode)
        self.currentState = self.env.reset(seed=np.random.randint(0, 100) if seed == None else seed)[0]

    def doAction(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation == 36 and observation != self.currentState:
            terminated = True
        if action == 1 and observation != self.currentState:
            reward = -.05
        self.currentState = observation
        return reward, terminated, truncated, info
    
    def GetCurrentState(self):
        return self.currentState
    
    def getPossibleActions(self):
        return [0, 1, 2, 3]

    def reset(self):
        pass

    def render(self):
        self.env.render()
        
