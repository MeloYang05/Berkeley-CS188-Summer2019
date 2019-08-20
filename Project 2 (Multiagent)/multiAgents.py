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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        Score=0

        #The number of food will influence the score
        #More the food, lower the score
        FoodNum=len(newFood.asList())
        Score-=FoodNum

        #The max distance to the Food will influence the score
        #More the distance, lower the score
        Fooddistance_list=list()
        for foodPos in newFood.asList():
            distance=manhattanDistance(foodPos,newPos)
            Fooddistance_list.append(distance)
        if len(Fooddistance_list)>0:
            Maxdistance_toFood=max(Fooddistance_list)
            Score-=Maxdistance_toFood

        #The min distance to the Ghost will influence the score
        #More the distance, higher the score
        Ghostdistance_list=list()
        for state in newGhostStates:
            distance=manhattanDistance(state.configuration.pos,newPos)
            Ghostdistance_list.append(distance)
        if len(Ghostdistance_list)>0:
            Mindistance_toGhost=min(Ghostdistance_list)
            Score+=Mindistance_toGhost
        
        #The scared time of the min distance Ghost will influence the score
        #Min the scared time, higher the score
        Score-=min(newScaredTimes)
        return Score+currentGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"
        myDepth=0
        myAgentIndex=0
        Action_Value=self.StateValue(gameState,myAgentIndex,myDepth)
        return Action_Value[0]

    def StateValue(self,gameState,myAgentIndex,myDepth):
        if myAgentIndex >= gameState.getNumAgents():
            myAgentIndex=0
            myDepth+=1
        if myDepth==self.depth:
            return self.evaluationFunction(gameState)
        if myAgentIndex == 0:
            return self.get_MaxValue(gameState,myAgentIndex,myDepth)
        else:
            return self.get_MinValue(gameState,myAgentIndex,myDepth)

    def get_MaxValue(self,gameState,myAgentIndex,myDepth):
        Now_Action_Value=("action",-float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a=="Stop":
                continue
            Predict_action_value=self.StateValue(gameState.generateSuccessor(myAgentIndex, a),myAgentIndex+1,myDepth)
            try:
                Predict_value=Predict_action_value[1]
            except:
                Predict_value=Predict_action_value
            if Predict_value>Now_Action_Value[1]:
                Now_Action_Value=(a,Predict_value)
        return Now_Action_Value
        

    def get_MinValue(self,gameState,myAgentIndex,myDepth):
        Now_Action_Value=("action",float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a=="Stop":
                continue
            Predict_action_value=self.StateValue(gameState.generateSuccessor(myAgentIndex, a),myAgentIndex+1,myDepth)
            try:
                Predict_value=Predict_action_value[1]
            except:
                Predict_value=Predict_action_value
            if Predict_value<Now_Action_Value[1]:
                Now_Action_Value=(a,Predict_value)
        return Now_Action_Value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        myDepth=0
        myAgentIndex=0
        Alpha=-float("inf")
        Beta=float("inf")
        Action_Value=self.StateValue(gameState,myAgentIndex,myDepth,Alpha,Beta)
        return Action_Value[0]

    def StateValue(self,gameState,myAgentIndex,myDepth,Alpha,Beta):
        if myAgentIndex >= gameState.getNumAgents():
            myAgentIndex=0
            myDepth+=1
        if myDepth==self.depth:
            return self.evaluationFunction(gameState)
        if myAgentIndex == 0:
            return self.get_MaxValue(gameState,myAgentIndex,myDepth,Alpha,Beta)
        else:
            return self.get_MinValue(gameState,myAgentIndex,myDepth,Alpha,Beta)
    
    def get_MaxValue(self,gameState,myAgentIndex,myDepth,Alpha,Beta):
        Now_Action_Value=("action",-float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a=="Stop":
                continue
            Predict_action_value=self.StateValue(gameState.generateSuccessor(myAgentIndex, a),myAgentIndex+1,myDepth,Alpha,Beta)
            try:
                Predict_value=Predict_action_value[1]
            except:
                Predict_value=Predict_action_value
            if Predict_value>Now_Action_Value[1]:
                Now_Action_Value=(a,Predict_value)
            if Predict_value>Beta:
                break
            if Predict_value>Alpha:
                Alpha=Predict_value
        return Now_Action_Value

    def get_MinValue(self,gameState,myAgentIndex,myDepth,Alpha,Beta):
        Now_Action_Value=("action",float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a=="Stop":
                continue
            Predict_action_value=self.StateValue(gameState.generateSuccessor(myAgentIndex, a),myAgentIndex+1,myDepth,Alpha,Beta)
            try:
                Predict_value=Predict_action_value[1]
            except:
                Predict_value=Predict_action_value
            if Predict_value<Now_Action_Value[1]:
                Now_Action_Value=(a,Predict_value)
            if Predict_value<Alpha:
                break
            if Predict_value<Beta:
                Beta=Predict_value
        return Now_Action_Value


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
        myDepth=0
        myAgentIndex=0
        Action_Value=self.StateValue(gameState,myAgentIndex,myDepth)
        return Action_Value[0]

    def StateValue(self,gameState,myAgentIndex,myDepth):
        if myAgentIndex >= gameState.getNumAgents():
            myAgentIndex=0
            myDepth+=1
        if myDepth==self.depth:
            return self.evaluationFunction(gameState)
        if myAgentIndex == 0:
            return self.get_MaxValue(gameState,myAgentIndex,myDepth)
        else:
            return self.get_ExpValue(gameState,myAgentIndex,myDepth)
    
    def get_MaxValue(self,gameState,myAgentIndex,myDepth):
        Now_Action_Value=("action",-float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a=="Stop":
                continue
            Predict_action_value=self.StateValue(gameState.generateSuccessor(myAgentIndex, a),myAgentIndex+1,myDepth)
            try:
                Predict_value=Predict_action_value[1]
            except:
                Predict_value=Predict_action_value
            if Predict_value>Now_Action_Value[1]:
                Now_Action_Value=(a,Predict_value)
        return Now_Action_Value

    def get_ExpValue(self,gameState,myAgentIndex,myDepth):
        Expected_Value=0
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)

        Probability=1/len(gameState.getLegalActions(myAgentIndex))

        for a in gameState.getLegalActions(myAgentIndex):
            if a=="Stop":
                continue
            Predict_action_value=self.StateValue(gameState.generateSuccessor(myAgentIndex, a),myAgentIndex+1,myDepth)
            try:
                Predict_value=Predict_action_value[1]
            except:
                Predict_value=Predict_action_value
            Expected_Value+=Predict_value*Probability
        Now_Action_Value=("action",Expected_Value)
        return Now_Action_Value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    Score=0
    Pacman_Pos=currentGameState.getPacmanPosition()

    Current_food=currentGameState.getFood().asList()
    FoodNum=len(Current_food)
    Score-=FoodNum

    Fooddistance_list=list()
    for foodPos in Current_food:
        distance=manhattanDistance(foodPos,Pacman_Pos)
        Fooddistance_list.append(distance)
    if len(Fooddistance_list)>0:
        Maxdistance_toFood=max(Fooddistance_list)
        Score-=Maxdistance_toFood

    GhostStates=currentGameState.getGhostStates()
    Ghostdistance_list=list()
    for state in GhostStates:
        distance=manhattanDistance(state.configuration.pos,Pacman_Pos)
        Ghostdistance_list.append(distance)
    if len(Ghostdistance_list)>0:
        Mindistance_toGhost=min(Ghostdistance_list)
        Score+=Mindistance_toGhost
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Score-=min(ScaredTimes)

    return Score+currentGameState.getScore()
    
    
# Abbreviation
better = betterEvaluationFunction
