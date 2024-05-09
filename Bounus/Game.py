import numpy as np
from copy import deepcopy
class GameState:
    def __init__(self, boardSize = 8) -> None:
        self.boardSize = boardSize
        self.board = [['-' for i in range(self.boardSize)] for j in range(self.boardSize)]
        self.board[3][3] = 'W'
        self.board[4][3] = 'B'
        self.board[3][4] = 'B'
        self.board[4][4] = 'W'

    def GetPieces(self, piece):
        result = []
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.board[i][j] == piece:
                    result.append((i, j))
        return result
    
    def IsEmpty(self, location):
        return self.board[location[0]][location[1]] == '-'

    def GetEmptyNeighbors(self, piece):
        result = []
        i = piece[0]
        j = piece[1]
        if j != self.boardSize - 1 and self.IsEmpty((i, j + 1)):
            result.append((i, j + 1))
        if j != 0 and self.IsEmpty((i, j - 1)):
            result.append((i, j - 1))
        if i != self.boardSize - 1 and self.IsEmpty((i + 1, j)):
            result.append((i + 1, j))
        if i != 0 and self.IsEmpty((i - 1, j)):
            result.append((i - 1, j))
        if i != self.boardSize - 1 and j != self.boardSize - 1 and self.IsEmpty((i + 1, j + 1)):
            result.append((i + 1, j + 1))
        if i != 0 and j != 0 and self.IsEmpty((i - 1, j - 1)):
            result.append((i - 1, j - 1))
        if i != self.boardSize - 1 and j != 0 and self.IsEmpty((i + 1, j - 1)):
            result.append((i + 1, j - 1))
        if i != 0 and j != self.boardSize - 1 and self.IsEmpty((i - 1, j + 1)):
            result.append((i - 1, j + 1))
        return result

    def GetOpponent(self, player):
        return 'W' if player == 'B' else 'B'

    def GetSuccessorInDirection(self, location, direction):
        if direction == 0:
            return (location[0] + 1, location[1] + 1)
        elif direction == 1:
            return (location[0] - 1, location[1] - 1)
        elif direction == 2:
            return (location[0] + 1, location[1])
        elif direction == 3:
            return (location[0] - 1, location[1])
        elif direction == 4:
            return (location[0], location[1] + 1)
        elif direction == 5:
            return (location[0], location[1] - 1)
        elif direction == 6:
            return (location[0] + 1, location[1] - 1)
        elif direction == 7:
            return (location[0] - 1, location[1] + 1)
    
    def IsAtBorderInDirection(self, location, direction):
        i = location[0]
        j = location[1]
        upperlimit = self.boardSize - 1
        if direction == 0:
            return i == upperlimit or j == upperlimit
        elif direction == 1:
            return i == 0 or j == 0
        elif direction == 2:
            return i == upperlimit
        elif direction == 3:
            return i == 0
        elif direction == 4:
            return j == upperlimit
        elif direction == 5:
            return j == 0
        elif direction == 6:
            return i == upperlimit or j == 0
        elif direction == 7:
            return i == 0 or j == upperlimit
        
    def GetFlippedIfPlaced(self, location, player):
        Flipped = set()
        opponent = self.GetOpponent(player)
        for direction in range(8):
            current = location
            localFlipped = set()
            #print(f"****Starting on {location} with direction {direction}*****")
            while True:
                if current == location:
                    if self.IsAtBorderInDirection(current, direction):
                        break
                    localFlipped.add(current)
                    current = self.GetSuccessorInDirection(current, direction)
                    continue
                #print(current)
                if self.board[current[0]][current[1]] == player:
                    break
                if self.IsAtBorderInDirection(current, direction) or self.IsEmpty(current):
                    localFlipped.clear()
                    break
                if self.board[current[0]][current[1]] == opponent:
                    localFlipped.add(current)
                    current = self.GetSuccessorInDirection(current, direction)
            [Flipped.add(piece) for piece in localFlipped]
        if location in Flipped:
            Flipped.remove(location)
        return Flipped
  
    def GetLegalActions(self, player):
        opponent = self.GetOpponent(player)
        opponentPieces = self.GetPieces(opponent)
        result = set()
        for piece in opponentPieces:
            neighboringPieces = self.GetEmptyNeighbors(piece)
            for candidate in neighboringPieces:
                if candidate not in result and len(self.GetFlippedIfPlaced(candidate, player)) != 0:
                    result.add(candidate)
        return result
    
    def Flip(self, location):
        if self.board[location[0]][location[1]] == 'W':
            self.board[location[0]][location[1]] = 'B'
        else:
            self.board[location[0]][location[1]] = 'W'

    def GenerateSuccessor(self, player, location):
        flippedPieces = self.GetFlippedIfPlaced(location, player)
        if len(flippedPieces) == 0:
            raise Exception(f"You cannot Place a piece in {location}")
        newGameState = deepcopy(self)
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if (i, j) in flippedPieces:
                    newGameState.Flip((i, j))
        newGameState.board[location[0]][location[1]] = player
        return newGameState
    
    def isTerminal(self):
        return len(self.GetLegalActions('W')) == 0 and len(self.GetLegalActions('B')) == 0

    def __str__(self) -> str:
        result = "   "
        for i in range(self.boardSize):
            result += f"{i}" + " "
        result += "\n"
        for i in range(self.boardSize):
            result += f"{i}" + "  "
            for j in range(self.boardSize):
                result += self.board[i][j] + " "
            result += "\n"
        return result




