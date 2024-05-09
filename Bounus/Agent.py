import math
from Game import GameState
class AlphaBetaAgent:
    def __init__(self, player, depth = 2):
        self.player = player
        self.depth = depth

    def getAction(self, gameState: GameState):
        bestMove = self.getMaxValue(gameState, self.player, self.depth, -math.inf, math.inf)
        return bestMove[0]

    def getMinValue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        if gameState.isTerminal() or depth == 0:
            return (None, self.evaluationFunction(gameState))
        nextAgent = gameState.GetOpponent(agentIndex)
        if nextAgent == self.player:
            depth -= 1
        legalMoves = gameState.GetLegalActions(agentIndex)
        if len(legalMoves) == 0:
            return (None, self.getMaxValue(gameState, nextAgent, depth, alpha, beta)[1])
        bestValue = math.inf
        bestAction = 0 
        options = [(action, self.evaluationFunction(gameState.GenerateSuccessor(agentIndex, action))) for action in legalMoves]
        options.sort(key=lambda x: x[1])
        legalMoves = [action for action, value in options]
        for action in legalMoves:
            newState = gameState.GenerateSuccessor(agentIndex, action)
            newValue = self.getMaxValue(newState, nextAgent, depth, alpha, beta)[1]
            if bestValue > newValue:
                bestValue = newValue
                bestAction = action
            if newValue < alpha:
                return (bestAction, bestValue)
            beta = min(beta, newValue)
        return (bestAction, bestValue)
    
    def getMaxValue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        if gameState.isTerminal() or depth == 0:
            return (None, self.evaluationFunction(gameState))
        legalMoves = gameState.GetLegalActions(agentIndex)
        nextAgent = gameState.GetOpponent(agentIndex)
        if len(legalMoves) == 0:
            return (None, self.getMinValue(gameState, nextAgent, depth, alpha, beta)[1])
        bestValue = -math.inf
        bestAction = 0 
        options = [(action, self.evaluationFunction(gameState.GenerateSuccessor(agentIndex, action))) for action in legalMoves]
        options.sort(key=lambda x: x[1], reverse=True)
        legalMoves = [action for action, value in options]
        for action in legalMoves:
            newState = gameState.GenerateSuccessor(agentIndex, action)
            newValue = self.getMinValue(newState, nextAgent, depth, alpha, beta)[1]
            if bestValue < newValue:
                bestValue = newValue
                bestAction = action
            if newValue > beta:
                return (bestAction, bestValue)
            alpha = max(alpha, newValue)
        return (bestAction, bestValue)

    def evaluationFunction(self, gamestate: GameState):
        return len(gamestate.GetPieces(self.player))
