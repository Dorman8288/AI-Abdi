import gymnasium as gym
import numpy as np
import keyboard
from time import sleep
class LunarKeyboardAgent:
    def __init__(self, FrameRate) -> None:
        self.FrameRate = FrameRate
        self.env = gym.make("LunarLander-v2", render_mode="human")
    def run(self):
        TotalReward = 0
        CurrentObservation, info = self.env.reset(seed=np.random.randint(1000))
        while True:
            if keyboard.is_pressed('esc'):
                break
            actions = set()
            for _ in range(50):
                if keyboard.is_pressed('a'):
                    actions.add(1)
                if keyboard.is_pressed('d'):
                    actions.add(3)
                if keyboard.is_pressed('w'):
                    actions.add(2)
            if len(actions) == 0:
                observation, reward, terminated, truncated, info = self.env.step(0)
                if terminated or truncated:
                    break
                TotalReward += reward
            for action in actions:
                TotalReward += reward
                if terminated or truncated:
                    break
                observation, reward, terminated, truncated, info = self.env.step(action)
            #print(action)
            sleep(1 / self.FrameRate)
        print(f"Total Reward Accuaired: {TotalReward}")


class CartPoleKeyboardAgent:
    def __init__(self, FrameRate) -> None:
        self.FrameRate = FrameRate
        self.env = gym.make("CartPole-v1", render_mode="human")
    def run(self):
        TotalReward = 0
        CurrentObservation, info = self.env.reset(seed=np.random.randint(1000))
        while True:
            if keyboard.is_pressed('esc'):
                break
            actions = set()
            for _ in range(50):
                if keyboard.is_pressed('a'):
                    actions.add(0)
                if keyboard.is_pressed('d'):
                    actions.add(1)
            over = False
            for action in actions:
                observation, reward, terminated, truncated, info = self.env.step(action)
                TotalReward += reward
                if truncated or terminated:
                    over = True
                    break
            if over:
                break
            #print(action)
            sleep(1 / self.FrameRate)
        print(f"Total Reward Accuaired: {TotalReward}")
            

class CliffWalkerKeyboardAgent:
    def __init__(self, FrameRate) -> None:
        self.FrameRate = FrameRate
        self.env = gym.make("CliffWalking-v0", render_mode="human")
    def run(self):
        TotalReward = 0
        CurrentObservation, info = self.env.reset(seed=np.random.randint(1000))
        while True:
            if keyboard.is_pressed('esc'):
                break
            actions = set()
            for _ in range(50):
                if keyboard.is_pressed('w'):
                    actions.add(0)
                if keyboard.is_pressed('d'):
                    actions.add(1)
                if keyboard.is_pressed('s'):
                    actions.add(2)
                if keyboard.is_pressed('a'):
                    actions.add(3)
            over = False
            for action in actions:
                observation, reward, terminated, truncated, info = self.env.step(action)
                TotalReward += reward
                if truncated or terminated:
                    over = True
                    break
            if over:
                break
            #print(action)
            sleep(1 / self.FrameRate)
        print(f"Total Reward Accuaired: {TotalReward}")
            