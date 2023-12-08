import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math


class IP:
    def __init__(self, name, p1=[0,0], p2=[0,0]):
        self.name = name
        self.p1 = p1
        self.p2 = p2

    def __copy__(self):
        return IP(self.name, self.p1[:], self.p2[:])

    def executeRound(self):
        result = [0, 0]
        # p1 attack
        ap = self.p1[0]
        dp = self.p2[1] * 2
        diff = ap - dp
        if (diff >= 0):
            # Attack win
            self.p2[1] = 0
            result[0] += diff
        else:
            # Def win
            self.p2[1] -= math.ceil(ap / 2)
        # p2 attack
        ap = self.p2[0]
        dp = self.p1[1] * 2
        diff = ap - dp
        if (diff >= 0):
            # Attack win
            self.p1[1] = 0
            result[1] += diff
        else:
            # Def win
            self.p1[1] -= math.ceil(ap / 2)

        self.p2[0] = 0
        self.p1[0] = 0
        # Returns tuple of (p1SuccessfulAttacks, p2SuccessfulAttacks)
        return result


class Game:
    def __init__(self, P=10, S=3, IPs=None):
        if IPs is None:
            IPs = [IP("Healthcare"), IP("Economics"), IP("Education")]
        self.P = P
        self.S = S # moves per turn
        self.IPs = IPs
        self.P1Power = P
        self.P2Power = P

    def __copy__(self):
        ips = []
        for ip in self.IPs:
            ips.append(ip.__copy__())
        g = Game(self.P, self.S, ips)
        g.P1Power = self.P1Power
        g.P2Power = self.P2Power
        return g

    def __executeRound(self):
        p1SuccAttack = 0
        p2SuccAttack = 0
        for ip in self.IPs:
            res = ip.executeRound()
            p1SuccAttack += res[0]
            p2SuccAttack += res[1]
        return (p1SuccAttack, p2SuccAttack)

    def playRound(self, p1Move, p2Move):
        # p1Move and p2Move are arrays of tuples with length Ips.length
        for i in range(len(self.IPs)):
            self.IPs[i].p1[0] += p1Move[i][0]
            self.IPs[i].p1[1] += p1Move[i][1]
            self.P1Power -= p1Move[i][0] + p1Move[i][1]
            self.IPs[i].p2[0] += p2Move[i][0]
            self.IPs[i].p2[1] += p2Move[i][1]
            self.P2Power -= p2Move[i][0] + p2Move[i][1]
        result = self.__executeRound()
        self.P1Power -= result[1] - result[0] # Own attacks come back!
        self.P2Power -= result[0] - result[1]

    def isWon(self):
        # 0 not won, 1 p1 won, 2 p2 won, 3 draw
        return 1 * (self.P2Power <= 0) + 2 * (self.P1Power <= 0)

    def _genPMoves(self, numMoves):
        # Generate all possible combinations using itertools.product
        all_combinations = list(product(range(numMoves + 1), repeat=2*len(self.IPs)))

        # Filter combinations where the sum is equal to n
        valid_combinations = [combo for combo in all_combinations if sum(combo) == numMoves]

        return valid_combinations

    def genAllPMoves(self, max):
        moves = []
        for i in range(1, max+1):
            moves.extend(self._genPMoves(i))

        toRet = []
        for mov in moves:
            result = []
            for i in range(int(len(mov) / 2)):
                result.append((mov[i*2], mov[(i*2)+1]))
            toRet.append(result)
        return toRet

    def genAllMoves(self):
        p1Moves = self.genAllPMoves(min(self.P1Power, self.S))
        p2Moves = self.genAllPMoves(min(self.P2Power, self.S))
        return list(product(p1Moves, p2Moves))

class MaxN:
    # For optimising P1
    def __init__(self, game):
        self.game = game

    def evaluate(self, game):
        won = game.isWon()
        if won <= 0:
            return game.P1Power - game.P2Power
        elif won <= 1:
            return 100
        elif won <= 2:
            return -100
        else:
            return 0

    def maxn(self, depth, maxPlayer, game):
        # Returns via recursion the value / gain of a game
        if (depth <= 0 or game.isWon() != 0):
            return self.evaluate(game)

        playerPower = game.P1Power if not maxPlayer else game.P2Power
        otherPlayerPower = game.P1Power if maxPlayer else game.P2Power

        # Handle 1 power case
        if (game.P1Power <= 1 and game.P2Power <= 1):
            return 0
        elif (game.P1Power <= 1):
            return -100
        elif (game.P2Power <= 1):
            return 100

        gain = float('-inf') if maxPlayer else float("inf")
        pmoves = game.genAllPMoves(min(playerPower - 1, game.S))

        #if (depth >= 3):
        #    print("Calculating maxn of ", depth, len(pmoves))

        for p1_move in pmoves:
            sum = 0
            n = 0
            for p2_move in game.genAllPMoves(min(otherPlayerPower - 1, game.S)):
                # Make a copy of the game to simulate the moves
                temp_game = game.__copy__()

                # Simulate the round with the current moves
                temp_game.playRound(p1_move, p2_move)

                # Calculate the gain for player 1
                sum += self.maxn(depth - 1, not maxPlayer, temp_game)
                n += 1

            # Update the gain if this move gives a better gain
            gain = max(gain, sum / n) if maxPlayer else min(gain, sum / n)

        return gain

    def findBestMove(self, depth):
        best_move = []
        max_gain = float('-inf')
        i = 0
        pMoves = self.game.genAllPMoves(min(self.game.P1Power - 1, self.game.S))
        for p1_move in pMoves:
            i += 1
            print(str(i) + " / " + str(len(pMoves)))
            sum = 0
            n = 0
            for p2_move in self.game.genAllPMoves(min(self.game.P2Power - 1, self.game.S)):
                # Make a copy of the game to simulate the moves
                temp_game = self.game.__copy__()

                # Simulate the round with the current moves
                temp_game.playRound(p1_move, p2_move)

                # Calculate the gain for player 1
                #sum += self.evaluate(temp_game)
                sum += self.maxn(depth, False, temp_game)
                n+=1

            # Update the best move if this move gives a higher gain
            if sum / n > max_gain:
                max_gain = sum / n
                best_move = [p1_move]
            elif (sum / n == max_gain):
                best_move.append(p1_move)
        return best_move


g = Game(5, 2)
m = MaxN(g)
print(m.findBestMove(6))
