# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    #Let's do the manhatten distance
    distance = []
    foodList = currentGameState.getFood().asList()
    pacmanPos = list(successorGameState.getPacmanPosition())

    if action == 'Stop':
        return -float("inf")

    for ghostState in newGhostStates:
        if ghostState.getPosition() == tuple(pacmanPos) and ghostState.scaredTimer is 0:
            return -float("inf") 

    for food in foodList:
        x = -1*abs(food[0] - pacmanPos[0])
        y = -1*abs(food[1] - pacmanPos[1])
        distance.append(x+y) 

    return max(distance)

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
    self.pacmanIndex = 0

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
          Directions.STOP:
            The stop direction, which is always legal
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState, currentAgentIndex, curDepth)
        print "Returning %s" % str(val)
        return val[0]

    def value(self, gameState, currentAgentIndex, curDepth): 
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else:
            return self.minValue(gameState, currentAgentIndex, curDepth)
        
    def minValue(self, gameState, currentAgentIndex, curDepth):
        v = ("unknown", float("inf"))
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = min(v[1], retVal)

            if vNew is not v[1]:
                v = (action, vNew) 
        
        #print "Returning minValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return v

    def maxValue(self, gameState, currentAgentIndex, curDepth):
        v = ("unknown", -1*float("inf"))
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = max(v[1], retVal)

            if vNew is not v[1]:
                v = (action, vNew) 
        
        #print "Returning maxValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        curDepth = 0
        currentAgentIndex = 0
        alpha = -1*float("inf")
        beta = float("inf")
        val = self.value(gameState, currentAgentIndex, curDepth, alpha, beta)
        print "Returning %s" % str(val)
        return val[0]

    def value(self, gameState, currentAgentIndex, curDepth, alpha, beta): 
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth, alpha, beta)
        else:
            return self.minValue(gameState, currentAgentIndex, curDepth, alpha, beta)
        
    def minValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
        v = ("unknown", float("inf"))
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth, alpha, beta)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = min(v[1], retVal)

            if vNew is not v[1]:
                v = (action, vNew) 
            
            if v[1] <= alpha:
                #print "Pruning with '%s' from min since alpha is %2.2f" % (str(v), alpha)
                return v
            
            beta = min(beta, v[1])
            #print "Setting beta to %2.2f" % beta
        
        #print "Returning minValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return v

    def maxValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
        v = ("unknown", -1*float("inf"))
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth, alpha, beta)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = max(v[1], retVal)

            if vNew is not v[1]:
                v = (action, vNew) 
            
            if v[1] >= beta:
                #print "Pruning with '%s' from min since beta is %2.2f" % (str(v), beta)
                return v

            alpha = max(alpha, v[1])
            #print "Setting alpha to %2.2f" % alpha

        #print "Returning maxValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return v


 
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          Directions.STOP:
            The stop direction, which is always legal
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState, currentAgentIndex, curDepth)
        print "Returning %s" % str(val)
        return val[0]

    def value(self, gameState, currentAgentIndex, curDepth): 
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else:
            return self.expValue(gameState, currentAgentIndex, curDepth)
        
    def expValue(self, gameState, currentAgentIndex, curDepth):
        v = ["unknown", 0]
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)
        
        prob = 1.0/len(gameState.getLegalActions(currentAgentIndex))
        
        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            v[1] += retVal * prob
            v[0] = action
        
        #print "Returning minValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return tuple(v)

    def maxValue(self, gameState, currentAgentIndex, curDepth):
        v = ("unknown", -1*float("inf"))
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = max(v[1], retVal)

            if vNew is not v[1]:
                v = (action, vNew) 
        
        #print "Returning maxValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return v
 

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: Evaluation is based on the 
    -> Distance to the nearest food
    -> Closest Ghost
    -> The current score
    -> How many capsules are on the board
    -> How many scared ghosts there are
    The goal is to try to increase the score while getting closer to other food pellets,
    staying away from the closestGhost (if possible) unless
    are scared, getting rid of capsules, and eating 
    scaredGhosts whenever possible
    """        
    "*** YOUR CODE HERE ***"
    #Let's do the manhatten distance
    distanceToFood = []
    distanceToNearestGhost = []
    distanceToCapsules = []
    score = 0

    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    numOfScaredGhosts = 0

    pacmanPos = list(currentGameState.getPacmanPosition())

    for ghostState in ghostStates:
        if ghostState.scaredTimer is 0:
            numOfScaredGhosts += 1
            distanceToNearestGhost.append(0)
            continue

        gCoord = ghostState.getPosition()
        x = abs(gCoord[0] - pacmanPos[0])
        y = abs(gCoord[1] - pacmanPos[1])
        if (x+y) == 0:
            distanceToNearestGhost.append(0)
        else:
            distanceToNearestGhost.append(-1.0/(x+y))

    for food in foodList:
        x = abs(food[0] - pacmanPos[0])
        y = abs(food[1] - pacmanPos[1])
        distanceToFood.append(-1*(x+y))  

    if not distanceToFood:
        distanceToFood.append(0)

    return max(distanceToFood) + min(distanceToNearestGhost) + currentGameState.getScore() - 100*len(capsuleList) - 20*(len(ghostStates) - numOfScaredGhosts)
 

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.
      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()