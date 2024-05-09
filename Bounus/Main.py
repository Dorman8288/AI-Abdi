from Game import GameState
from Agent import AlphaBetaAgent
import time
import os
import numpy as np

log = ""
print("Welcome to Othello!\nChoose Your Color:\n1- B (Black)\n2- W (White)")
player = 'W' if input() == "W" else 'B'
print("choose Difficaulty:\n1- E (Easy)\n2- M (Medium)\n3- H (Hard)")
difficaulty = input()
depth = 1 if difficaulty == "E" else 2 if difficaulty == "M" else 3
opponent = GameState.GetOpponent(None, player)
currentState = GameState()
AIAgent = AlphaBetaAgent(opponent, depth)
turn = 'B'
log += f"Player: {player}\nDificaulty: {difficaulty}\n"
while True:
    os.system('cls')
    print(currentState)
    log += str(currentState) + "\n"
    if len(currentState.GetLegalActions(player)) == 0 and len(currentState.GetLegalActions(opponent)) == 0:
        print("Game Is Finished")
        log += "Game Is Finished" + "\n"
        blackCount = len(currentState.GetPieces('B'))
        whiteCount = len(currentState.GetPieces('W'))
        print(f"Black Count: {blackCount}")
        print(f"White Count: {whiteCount}")
        print(f"{'Black' if blackCount > whiteCount else 'White'} Wins!")
        log += f"Black Count: {blackCount}" + "\n" + f"White Count: {whiteCount}" + "\n" + f"{'Black' if blackCount > whiteCount else 'White'} Wins!" + "\n"
        if os.path.exists("history.txt"):
            os.remove("history.txt")
        f = open("history.txt", "w")
        f.write(log)
        break
    if turn == player:
        actions = currentState.GetLegalActions(player)
        if len(actions) == 0:
            print("You Cannot Place")
            log += "You Cannot Place\n"
            turn = GameState.GetOpponent(None, turn)
            time.sleep(3)
            continue
        print("You Can Place in:")
        log += "You Can Place in:\n"
        for index, action in enumerate(actions):
            print(f"{index}- {action}")
            log += f"{index}- {action}" + "\n"
        while True:
            try:
                print("Enter x:")
                x = int(input())
                print("Enter y:")
                y = int(input())
                currentState = currentState.GenerateSuccessor(player, (x, y))
            except:
                print(f"You Cannot Place In {(x, y)}")
                log += f"You Cannot Place In {(x, y)}\n"
                continue
            else:
                break
        os.system('cls')
        print(currentState)
        log += str(currentState) + "\n" + f"You Placed In {(x, y)}\n"
        print(f"You Placed In {(x, y)}")
    else:
        AiAction = AIAgent.getAction(currentState)
        if AiAction == None:
            print("Opponent Cannot Place")
            log += "Opponent Cannot Place\n"
            turn = GameState.GetOpponent(None, turn)
            time.sleep(3)
            continue
        currentState = currentState.GenerateSuccessor(opponent, AiAction)
        os.system('cls')
        print(currentState)
        print(f"Opponent Placed In {AiAction}")
        log += str(currentState) + "\n" + f"Opponent Placed In {AiAction}\n"
    turn = GameState.GetOpponent(None, turn)
    time.sleep(3)
