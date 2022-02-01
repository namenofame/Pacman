# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        walls = successorGameState.getWalls()

        ghostDistSum = 1
        for ghostState in newGhostStates:
          ghostDistSum += abs(newPos[0] - ghostState.getPosition()[0]) + abs(newPos[1] - ghostState.getPosition()[1]) 

        foodDistSum = 1
        closestDist = 1e8
        numFood = 0
        for i in range(newFood.width):
          for j in range(newFood.height):
            if newFood.data[i][j]:
              numFood += 1
              dist = abs(newPos[0] - i) + abs(newPos[1] - j)
              foodDistSum += dist
              closestDist = min(closestDist, dist)

        closestDist2 = 0
        queue = []
        queue.append(newPos)
        dx = [1, -1, 0, 0]
        dy = [0, 0, -1, 1]
        dist = [[-1 for y in range(newFood.height)] for x in range(newFood.width)]
        dist[newPos[0]][newPos[1]] = 1
        done = False
        while(len(queue)):
          curPos = queue.pop(0)
          if newFood[curPos[0]][curPos[1]]:
                closestDist2 = dist[curPos[0]][curPos[1]]
                break
          for k in range(4):
            x = curPos[0] + dx[k]
            y = curPos[1] + dy[k]
            if x >= 0 and x < newFood.width and y >= 0 and y < newFood.height and not walls[x][y]:
                if dist[x][y] is -1:
                  dist[x][y] = dist[curPos[0]][curPos[1]] + 1
                  queue.append((x, y))

        score = - 2.0 / ghostDistSum - 1.0 * numFood 
        if closestDist2 > 0:
          score += 1.5 / closestDist2
        return score 

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        scores = [self.miniMaxSearch(gameState.generateSuccessor(0, action), 1, self.depth - 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    # defined by me
    def miniMaxSearch(self, gameState, agentIndex, curDepth):
      
      agentIndex = agentIndex % gameState.getNumAgents()

      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      
      if curDepth is 0 and agentIndex is 0:
        return self.evaluationFunction(gameState)


      if agentIndex is 0:
        legalMoves = gameState.getLegalActions(0)
        scores = [self.miniMaxSearch(gameState.generateSuccessor(agentIndex, action), 1, curDepth - 1) for action in legalMoves]
        return max(scores)
      else:
        legalMoves = gameState.getLegalActions(agentIndex)
        scores = [self.miniMaxSearch(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, curDepth) for action in legalMoves]
        return min(scores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        alpha = float('-inf')
        beta = float('inf')
        scores = []
        # I did processing for top node separately because we want action related to optimal path and if there are ties then select randomly so that game doesn't hang
        for action in legalMoves:
          curStateScore = float('-inf')
          nextStateScore = self.miniMaxSearch(gameState.generateSuccessor(0, action), 1, self.depth - 1, alpha, beta)
          curStateScore = max(curStateScore, nextStateScore)
          alpha = max(alpha, curStateScore)
          scores.append(nextStateScore)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    # defined by me
    def miniMaxSearch(self, gameState, agentIndex, curDepth, alpha, beta):
      # print gameState
      # print alpha
      # print beta
      agentIndex = agentIndex % gameState.getNumAgents()

      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      
      if curDepth is 0 and agentIndex is 0:
        return self.evaluationFunction(gameState)


      if agentIndex is 0:
        # max agent
        legalMoves = gameState.getLegalActions(0)
        stateScore = float('-inf')
        for action in legalMoves:
          nextStateScore = self.miniMaxSearch(gameState.generateSuccessor(agentIndex, action), 1, curDepth - 1, alpha, beta)
          stateScore = max(stateScore, nextStateScore)
          if stateScore > beta:
            return stateScore
          alpha = max(alpha, stateScore)
        return stateScore
      else:
        # min agent
        legalMoves = gameState.getLegalActions(agentIndex)
        stateScore = float('inf')
        for action in legalMoves:
          nextStateScore = self.miniMaxSearch(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, curDepth, alpha, beta)
          stateScore = min(stateScore, nextStateScore)
          if stateScore < alpha:
            return stateScore
          beta = min(beta, stateScore)
        return stateScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        scores = [self.expectiMaxSearch(gameState.generateSuccessor(0, action), 1, self.depth - 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    # defined by me
    def expectiMaxSearch(self, gameState, agentIndex, curDepth):
      
      agentIndex = agentIndex % gameState.getNumAgents()

      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      
      if curDepth is 0 and agentIndex is 0:
        return self.evaluationFunction(gameState)


      if agentIndex is 0:
        legalMoves = gameState.getLegalActions(0)
        scores = [self.expectiMaxSearch(gameState.generateSuccessor(agentIndex, action), 1, curDepth - 1) for action in legalMoves]
        return max(scores)
      else:
        legalMoves = gameState.getLegalActions(agentIndex)
        scores = [self.expectiMaxSearch(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, curDepth) for action in legalMoves]
        return float(sum(scores)) / float(len(legalMoves))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    walls = currentGameState.getWalls()

    newCapsules = currentGameState.getCapsules()

    isCapsuleNear = False
    legalMoves = currentGameState.getLegalActions(0)
    for action in legalMoves:
      nxtState = currentGameState.generateSuccessor(0, action)
      if newCapsules.count(nxtState.getPacmanPosition()):
        isCapsuleNear = True

    allGhostsScared = True
    totalScaredTime = 1.0
    for time in newScaredTimes:
      totalScaredTime += time
      if time < 2:
        allGhostsScared = False

    ghostDistSum = 1
    for ghostState in newGhostStates:
      ghostDistSum += abs(newPos[0] - ghostState.getPosition()[0]) + abs(newPos[1] - ghostState.getPosition()[1]) 

    foodDistSum = 1
    numFood = 0
    for i in range(newFood.width):
      for j in range(newFood.height):
        if newFood.data[i][j]:
          numFood += 1
          dist = abs(newPos[0] - i) + abs(newPos[1] - j)
          foodDistSum += dist

    closestDist2 = 0
    queue = []
    queue.append(newPos)
    dx = [1, -1, 0, 0]
    dy = [0, 0, -1, 1]
    dist = [[-1 for y in range(newFood.height)] for x in range(newFood.width)]
    dist[newPos[0]][newPos[1]] = 1
    done = False
    while(len(queue)):
      curPos = queue.pop(0)
      if newFood[curPos[0]][curPos[1]]:
            closestDist2 = dist[curPos[0]][curPos[1]]
            break
      for k in range(4):
        x = curPos[0] + dx[k]
        y = curPos[1] + dy[k]
        if x >= 0 and x < newFood.width and y >= 0 and y < newFood.height and not walls[x][y]:
            if dist[x][y] is -1:
              dist[x][y] = dist[curPos[0]][curPos[1]] + 1
              queue.append((x, y))

    score = - 1.0 * numFood 

    if not allGhostsScared and isCapsuleNear:
      score -= 1.5 / totalScaredTime 

    if allGhostsScared:
      score += 2.5 / ghostDistSum
    else:
      score -= 2.7 / ghostDistSum

    if closestDist2 > 0:
      score += 1.7 / closestDist2

    return score 

# Abbreviation
better = betterEvaluationFunction

