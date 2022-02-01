# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
            Your value iteration agent should take an mdp on
            construction, run the indicated number of iterations
            and then act according to the resulting policy.

            Some useful mdp methods you will use:
                mdp.getStates()
                mdp.getPossibleActions(state)
                mdp.getTransitionStatesAndProbs(state, action)
                mdp.getReward(state, action, nextState)
                mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        curIteration = 1
        while curIteration <= self.iterations:
            nxtIterValues = util.Counter()
            states = self.mdp.getStates()
            # print curIteration
            for state in states:
                maxValue = 0.0
                legalActions = self.mdp.getPossibleActions(state)
                if len(legalActions):
                    maxValue = float('-inf')
                    for action in legalActions:
                        value = self.computeQValueFromValues(state, action)
                        maxValue = max(maxValue, value)
                nxtIterValues[state] = maxValue
                # print state
                # print nxtIterValues[state]
                # print ""
            self.values = nxtIterValues
            curIteration += 1


    def getValue(self, state):
        """
            Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
            Compute the Q-value of action in state from the
            value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0.0
        for transitionState in transitionStatesAndProbs:
            reward = self.mdp.getReward(state, action, transitionState[0])
            qValue += transitionState[1] * (reward + self.discount * self.values[transitionState[0]])

        return qValue

    def computeActionFromValues(self, state):
        """
            The policy is the best action in the given state
            according to the values currently stored in self.values.

            You may break ties any way you see fit.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.mdp.getPossibleActions(state)
        if len(legalActions) is 0:
            return None
        
        maxValue = float('-inf')
        bestAction = legalActions[0]
        for action in legalActions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > maxValue:
                maxValue = qValue
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
            Your cyclic value iteration agent should take an mdp on
            construction, run the indicated number of iterations,
            and then act according to the resulting policy. Each iteration
            updates the value of only one state, which cycles through
            the states list. If the chosen state is terminal, nothing
            happens in that iteration.

            Some useful mdp methods you will use:
                mdp.getStates()
                mdp.getPossibleActions(state)
                mdp.getTransitionStatesAndProbs(state, action)
                mdp.getReward(state)
                mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        curIteration = 1
        curStateIndex = 0
        states = self.mdp.getStates()
        totalStates = len(states)

        while curIteration <= self.iterations:
            curState = states[curStateIndex]
            maxValue = 0.0
            legalActions = self.mdp.getPossibleActions(curState)
            if len(legalActions):
                maxValue = float('-inf')
                for action in legalActions:
                    value = self.computeQValueFromValues(curState, action)
                    maxValue = max(maxValue, value)
            self.values[curState] = maxValue
            curIteration += 1
            curStateIndex += 1
            curStateIndex %= totalStates

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
            Your prioritized sweeping value iteration agent should take an mdp on
            construction, run the indicated number of iterations,
            and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        statePredecessors = {}
        priorityQueue = util.PriorityQueue()
        states = self.mdp.getStates()
        for state in states:
            statePredecessors[state] = set()
            for candidateState in states:
                legalActions = self.mdp.getPossibleActions(candidateState)
                for action in legalActions:
                    transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(candidateState, action)
                    for transitionState in transitionStatesAndProbs:
                        if transitionState[1] > 0.0 and transitionState[0] == state:
                            statePredecessors[state].add(candidateState)
                            break
            if not self.mdp.isTerminal(state):
                diff = abs(self.computeNewValueForState(state) - self.values[state])
                priorityQueue.update(state, -diff)

        curIteration = 1
        while curIteration <= self.iterations:
            if priorityQueue.isEmpty():
                break
            state = priorityQueue.pop()
            newValue = self.computeNewValueForState(state)
            self.values[state] = newValue
            for predecessor in statePredecessors[state]:
                diff = abs(self.computeNewValueForState(predecessor) - self.values[predecessor])
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)

            curIteration += 1

    def computeNewValueForState(self, state):
        legalActions = self.mdp.getPossibleActions(state)
        maxValue = float('-inf')
        for action in legalActions:
            qValue = self.computeQValueFromValues(state, action)
            maxValue = max(maxValue, qValue)
        return maxValue






