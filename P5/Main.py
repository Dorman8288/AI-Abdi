import gymnasium as gym
from KeyboardAgents import *
from Enviroments import WalkingCliffEnviroment, CartPoleEnviroment, LunarLanderEnviroment
from InteligentAgents import QlearningAgent, ApproximateQAgent
from featureExtractors import CartPoleExtractor, LunarLanderExtractor
import math
discount = 0.3
EpisodeCount = 1000

epsilon = 0.9
epsilonRate = 0.9
alpha = 0.2

# env = WalkingCliffEnviroment(None, True)
# agent = QlearningAgent(env)
# agent.Train(EpisodeCount, discount, alpha, epsilon, epsilonRate, False, 120)
# print(agent.qValues)
# agent.Save("CartPole_Qlearning")
# env = WalkingCliffEnviroment("human", True)
# agent.Test(3, 30, env )


# discount = 0.3
# EpisodeCount = 100000

# epsilon = 0.9
# epsilonRate = 0.9
# alpha = 0.01
# try:
#     env = CartPoleEnviroment(None, False)
#     agent = ApproximateQAgent(CartPoleExtractor(), env)
#     agent.Train(EpisodeCount, discount, alpha, epsilon, epsilonRate, False, 120)
#     print(agent.weights)
# finally:
#     agent.Save("CartPole_Approximate")

# testEnv = CartPoleEnviroment("human", True)
# agent = QlearningAgent(testEnv)
# agent.Load("CartPole_Qlearning")
# agent.Test(5, 30, testEnv)


testEnv = CartPoleEnviroment("human", False)
agent = ApproximateQAgent(CartPoleExtractor(), testEnv)
agent.Load("CartPole_Approximate")
agent.Test(5, 30, testEnv)

# discount = 0.3
# EpisodeCount = 1000

# epsilon = 0.9
# epsilonRate = 0.9
# alpha = 0.2

# env = LunarLanderEnviroment(None, False)
# agent = QlearningAgent(LunarLanderExtractor(), env)
# agent.Train(EpisodeCount, discount, alpha, epsilon, epsilonRate, None, 120)
# print(agent.weights)
# testEnv = LunarLanderEnviroment("human", False)
# agent.Test(100, 30, testEnv)


# env.close()