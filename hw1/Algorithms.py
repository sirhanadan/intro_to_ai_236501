from math import sqrt

import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self, state, father = None, action = None, cost = 0):
        self.state = state
        self.father = father
        self.action = action
        self.cost = cost
        self.g = cost
        self.h = 0
        self.f = 0



class BFSAgent():
    def __init__(self) -> None:
        self.env = None
        actionss = []
        expanded_num = 0
        cost = 0


    def traverse_back_for_BFS_sol(self, current_node, total_cost, actions_list, expanded)->Tuple[List[int], float, int]:
        #if the  node doesn't have a father: it is an  initial state - stop
        if current_node.father is None:
            return actions_list, total_cost, expanded
        #change the state on the board
        self.env.set_state(current_node.state)
        #insert at the beginning of the actions list because we are travesing back
        actions_list.insert(0, current_node.action)
        total_cost = total_cost + current_node.cost
        return self.traverse_back_for_BFS_sol(current_node.father, total_cost, actions_list, expanded)



    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        #reset the values for a new search
        self.expanded_num = 0
        self.cost = 0.0
        #create your own environment to be able to change and use it without altering the original
        self.env = env
        self.env.reset()
        #create a  queue for opened nodes
        opened = []
        #create a set for closed nodes (a set and not list as explained in tutorial 1
        closed = set()
        #define an initial state and create the first node in the graph
        initial_state = self.env.get_initial_state()
        initial_node = Node(initial_state, None, None, 0)
        #make sure the initial state is not a goal state
        if self.env.is_final_state(initial_state):
            return [0], 0.0, 0
        #add initial_node to opened
        opened.append(initial_node)
        #while the opened queue isn't empty (in python there isn't a size function but it returns false if it is empty):
        while(bool(opened)):
            current_node = opened.pop(0)
            #updating closed list happens here because otherise we might open this same node as one of its own children an infinite amount oif times
            self.expanded_num = self.expanded_num + 1
            closed.add(current_node.state)
            #if it doesn't have any successors move on
            """
            if None in self.env.succ(current_node.state):
                continue
            """
            self.env.set_state(current_node.state)
            #if it does have successors, add them to queue and check if goals
            if None not in self.env.succ(current_node.state)[1]:
                for suc_action, suc_state_cost_term in self.env.succ(current_node.state).items():
                    #self.env.set_state(current_node.state)
                    suc_state, suc_cost, suc_term = suc_state_cost_term
                    suc_node = Node(suc_state, current_node, suc_action, suc_cost)
                    #if the node has been closed or opened move on, otherwise open it
                    if suc_state in closed:
                        continue
                    if suc_state in [n.state for n in opened]:
                        continue
                    #check if goal state
                    if self.env.is_final_state(suc_state):
                        return self.traverse_back_for_BFS_sol(suc_node, self.cost, [], self.expanded_num)
                    opened.append(suc_node)
        #in case of error
        return [0], -1.0,-1




class DFSAgent():
    def __init__(self) -> None:
        self.env = None
        actionss = []
        expanded_num = 0
        cost = 0


    def traverse_back_for_DFS_sol(self, current_node, total_cost, actions_list, expanded)->Tuple[List[int], float, int]:
        #if the  node doesn't have a father: it is an  initial state - stop
        if current_node.father is None:
            return actions_list, total_cost, expanded
        #change the state on the board
        self.env.set_state(current_node.state)
        #insert at the beginning of the actions list because we are travesing back
        actions_list.insert(0, current_node.action)
        total_cost = total_cost + current_node.cost
        return self.traverse_back_for_DFS_sol(current_node.father, total_cost, actions_list, expanded)


    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        #reset the values for a new search
        self.expanded_num = 0
        self.cost = 0.0
        # create your own environment to be able to change and use it without altering the original
        self.env = env
        self.env.reset()
        # create a  queue for opened nodes
        opened = []
        # create a set for closed nodes (a set and not list as explained in tutorial 1
        closed = set()
        # define an initial state and create the first node in the graph
        initial_state = self.env.get_initial_state()
        initial_node = Node(initial_state, None, None, 0)
        # make sure the initial state is not a goal state
        if self.env.is_final_state(initial_state):
            return [0], 0.0, 0
        # add initial_node to opened
        opened.append(initial_node)
        return self.recursive_DFS_search(opened, closed)

    def recursive_DFS_search(self, opened, closed) -> Tuple[List[int], float, int]:
        #in DFS algorithm, opened is a stack and not a queue (as explained by Abdul Bari on youtube god bless him), therefore we pop the last element
        #if the stack is empty there is no solution
        if not bool(opened):
            return [0], -1.0, -1
        current_node = opened.pop()
        if self.env.is_final_state(current_node.state):
            """print("FOUND GOAL")"""
            return self.traverse_back_for_DFS_sol(current_node,self.cost,[],self.expanded_num)
        # updating closed list happens here because otherise we might open this same node as one of its own children an infinite amount oif times
        self.expanded_num = self.expanded_num + 1
        closed.add(current_node.state)
        #if it doesn't have any successor states (aka hole), terminate, you've got an unsolvable board
        if None in self.env.succ(current_node.state)[1]:
            """print("FOUND HOLE")
            d = self.env.succ(current_node.state)
            print(d)"""
            if not bool(opened):
                return [0], -1.0, -1
            return [0], -1.0, -1
        #set board's current state
        self.env.set_state(current_node.state)
        # if it does have successors, add them to queue and check if goals
        for suc_action, suc_state_cost_term in self.env.succ(current_node.state).items():
            self.env.set_state(current_node.state)
            suc_state, suc_cost, suc_term = suc_state_cost_term
            suc_node = Node(suc_state, current_node, suc_action, suc_cost)
            # if the node has been closed or opened move on, otherwise open it
            if suc_state in closed:
                continue
            if suc_state in [n.state for n in opened]:
                continue
            opened.append(suc_node)
            #move on to next son
            #it is possible that the algorithm might reach a hole, in which case the returned value (recursively) would be [0], -1.0, -1
            #in order to avoid such cases, when we reach a hole and the returned value is as described, dont return it, just forget it and move on to the next son
            res = self.recursive_DFS_search(opened, closed)
            if not res[1] == -1.0:
                return res
        # in case of error
        return [0], -1.0, -1






class UCSAgent():

    def __init__(self) -> None:
        self.env = None
        actionss = []
        expanded_num = 0
        cost = 0.0

    def traverse_back_for_UCS_sol(self, current_node, total_cost, actions_list, expanded)->Tuple[List[int], float, int]:
        #if the  node doesn't have a father: it is an  initial state - stop
        if current_node.father is None:
            return actions_list, total_cost, expanded
        #change the state on the board
        self.env.set_state(current_node.state)
        #insert at the beginning of the actions list because we are travesing back
        actions_list.insert(0, current_node.action)
        total_cost = total_cost + current_node.cost
        return self.traverse_back_for_UCS_sol(current_node.father, total_cost, actions_list, expanded)


    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        # reset the values for a new search
        self.expanded_num = 0
        self.cost = 0.0
        # create your own environment to be able to change and use it without altering the original
        self.env = env
        self.env.reset()
        # create a priority queue for opened nodes
        opened = heapdict.heapdict()
        # create a set for closed nodes (a set and not list as explained in tutorial 1
        closed = set()
        # define an initial state and create the first node in the graph
        initial_state = self.env.get_initial_state()
        initial_node = Node(initial_state, None, None, 0)
        # make sure the initial state is not a goal state
        if self.env.is_final_state(initial_state):
            return [0], 0.0, 0
        # add initial_node to opened (basically like adding to a dictionary
        #as requested in the notebook: keys=Nodes, values = a tuple of g(cost so far), state number (we also added the node itself)
        opened[initial_state] =  (0, initial_node.state,initial_node)
        while bool(opened):
            current_node = opened.popitem()[1][2]
            if self.env.is_final_state(current_node.state):
                return self.traverse_back_for_UCS_sol(current_node, self.cost, [], self.expanded_num)
            self.expanded_num= self.expanded_num + 1
            closed.add(current_node.state)
            self.env.set_state(current_node.state)
            # if it does have successors, add them to queue based on priority and check if goals
            if None not in self.env.succ(current_node.state)[1]:
                for suc_action, suc_state_cost_term in self.env.succ(current_node.state).items():
                    # self.env.set_state(current_node.state)
                    suc_state, suc_cost, suc_term = suc_state_cost_term
                    suc_node = Node(suc_state, current_node, suc_action, suc_cost)
                    # update new cost for the successors which is their own cost plus the cost of everything up until now which is basically g of the father node
                    suc_node.g = suc_node.g + current_node.g
                    # if the node has been closed move on
                    if suc_state in closed:
                        continue
                    #if the node hasn't been opened yet open it
                    if not suc_state in [s for s in opened] :
                        opened[suc_state] = (suc_node.g, suc_state, suc_node)
                    #if the node has been opened but the new g is less than the g it had been originally inserted with, replace it
                    if suc_state in [s for s in opened.keys()]:
                        old_node_g = opened[suc_state][0]
                        if suc_node.g < old_node_g:
                            opened[suc_state] = (suc_node.g, suc_state, suc_node)
                    #the check if node is goal state IN UCS IS done IN EXPANSION TIME
                    #if self.env.is_final_state(suc_state):
                    #    return self.traverse_back_for_UCS_sol(suc_node, self.cost, [], 0)
            # in case of error
        return [0], -1.0, -1




class GreedyAgent():
  
    def __init__(self) -> None:
        self.env = None
        actionss = []
        expanded_num = 0
        cost = 0

    def h_sap_heuristic(self, state_num, N):
        row , col = self.env.to_row_col(state_num)
        manhattan_distance = (N-row) + (N-col)
        cost_p = 100
        return min(manhattan_distance, cost_p)

    def traverse_back_for_GA_sol(self, current_node, total_cost, actions_list, expanded)->Tuple[List[int], float, int]:
        #if the  node doesn't have a father: it is an  initial state - stop
        if current_node.father is None:
            return actions_list, total_cost, expanded
        #change the state on the board
        self.env.set_state(current_node.state)
        #insert at the beginning of the actions list because we are travesing back
        actions_list.insert(0, current_node.action)
        total_cost = total_cost + current_node.cost
        return self.traverse_back_for_GA_sol(current_node.father, total_cost, actions_list, expanded)

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        # reset the values for a new search
        self.expanded_num = 0
        self.cost = 0.0
        # create your own environment to be able to change and use it without altering the original
        self.env = env
        self.env.reset()
        #size of the board
        N = sqrt(self.env.observation_space.n)
        # create a priority queue for opened nodes
        opened = heapdict.heapdict()
        # create a set for closed nodes (a set and not list as explained in tutorial 1
        closed = set()
        # define an initial state and create the first node in the graph
        initial_state = self.env.get_initial_state()
        initial_node = Node(initial_state, None, None, 0)
        initial_node.h = self.h_sap_heuristic(initial_state, N)
        # make sure the initial state is not a goal state
        if self.env.is_final_state(initial_state):
            return [0], 0.0, 0
        # add initial_node to opened (basically like adding to a dictionary
        # as requested in the notebook: keys=Nodes, values = a tuple of cost and state number
        opened[initial_state] = (initial_node.h, initial_node.state, initial_node)
        while bool(opened):
            current_node = opened.popitem()[1][2]
            if self.env.is_final_state(current_node.state):
                return self.traverse_back_for_GA_sol(current_node, self.cost, [], self.expanded_num)
            self.expanded_num = self.expanded_num + 1
            closed.add(current_node.state)
            self.env.set_state(current_node.state)
            # if it does have successors, add them to queue based on priority and check if goals
            if None not in self.env.succ(current_node.state)[1]:
                for suc_action, suc_state_cost_term in self.env.succ(current_node.state).items():
                    # self.env.set_state(current_node.state)
                    suc_state, suc_cost, suc_term = suc_state_cost_term
                    suc_node = Node(suc_state, current_node, suc_action, suc_cost)
                    # update new cost for the successors which is their own cost plus the cost of everything up until now which is basically g of the father node
                    suc_node.h = self.h_sap_heuristic(suc_state, N)
                    # if the node has been closed move on
                    if suc_state in closed:
                        continue
                    # if the node hasn't been opened yet open it
                    if not suc_state in [s for s in opened]:
                        opened[suc_state] = (suc_node.h, suc_state, suc_node)
                    # if the node has been opened but the new h is less than the h it had been originally inserted with, replace it
                    if suc_state in [s for s in opened.keys()]:
                        old_node_h = opened[suc_state][0]
                        if suc_node.h < old_node_h:
                            opened[suc_state] = (suc_node.h, suc_state, suc_node)

                    #the check if node is goal state IN GBFS IS done IN EXPANSION TIME# check if goal state'''
                    # if self.env.is_final_state(suc_state):
                    #    return self.traverse_back_for_UCS_sol(suc_node, self.cost, [], 0)
            # in case of error
        return [0], -1.0, -1

class WeightedAStarAgent():

    def __init__(self):
        self.env = None
        actionss = []
        expanded_num = 0
        cost = 0


    def manhattan_distance_for_multiple_goals(self, state_num, N):
        min = 100
        row, col = self.env.to_row_col(state_num)
        for goal in self.env.goals:
            g_row , g_col = self.env.to_row_col(goal)
            manhattan_distance = (g_row - row) + (g_col - col)
            if manhattan_distance <= min:
                min = manhattan_distance
        return min
    def h_sap_heuristic(self, state_num, N):
        #row , col = self.env.to_row_col(state_num)
        manhattan_distance = self.manhattan_distance_for_multiple_goals(state_num,N)
        cost_p = 100
        return min(manhattan_distance, cost_p)




    def traverse_back_for_WAS_sol(self, current_node, total_cost, actions_list, expanded)->Tuple[List[int], float, int]:
        #if the  node doesn't have a father: it is an  initial state - stop
        if current_node.father is None:
            return actions_list, total_cost, expanded
        #change the state on the board
        self.env.set_state(current_node.state)
        #insert at the beginning of the actions list because we are travesing back
        actions_list.insert(0, current_node.action)
        total_cost = total_cost + current_node.cost
        return self.traverse_back_for_WAS_sol(current_node.father, total_cost, actions_list, expanded)

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], float, int]:
        # reset the values for a new search
        self.expanded_num = 0
        self.cost = 0.0
        # create your own environment to be able to change and use it without altering the original
        self.env = env
        self.env.reset()
        # size of the board
        N = sqrt(self.env.observation_space.n)
        # create a priority queue for opened nodes
        opened = heapdict.heapdict()
        # create a set for closed nodes (a set and not list as explained in tutorial 1
        closed = {}
        # define an initial state and create the first node in the graph
        initial_state = self.env.get_initial_state()
        initial_node = Node(initial_state, None, None, 0)
        initial_node.h = self.h_sap_heuristic(initial_state, N)
        initial_node.f = (h_weight*initial_node.h) + (1-h_weight)*initial_node.g
        #initial_node.f = (h_weight * initial_node.h) + initial_node.g
        # make sure the initial state is not a goal state
        if self.env.is_final_state(initial_state):
            return [0], 0.0, 0
        # add initial_node to opened (basically like adding to a dictionary
        # as requested in the notebook: keys=Nodes, values = a tuple of cost and state number
        opened[initial_state] = (initial_node.f, initial_node.state, initial_node)
        while bool(opened):
            current_node = opened.popitem()[1][2]
            if self.env.is_final_state(current_node.state):
                return self.traverse_back_for_WAS_sol(current_node, self.cost, [], self.expanded_num)
            self.expanded_num = self.expanded_num + 1
            closed[current_node.state] = current_node
            self.env.set_state(current_node.state)
            # if it does have successors, add them to queue based on priority and check if goals
            if None not in self.env.succ(current_node.state)[1]:
                for suc_action, suc_state_cost_term in self.env.succ(current_node.state).items():
                    # self.env.set_state(current_node.state)
                    suc_state, suc_cost, suc_term = suc_state_cost_term
                    suc_node = Node(suc_state, current_node, suc_action, suc_cost)
                    # update new cost for the successors which is their own cost plus the cost of everything up until now which is basically g of the father node
                    suc_node.h = self.h_sap_heuristic(suc_state, N)
                    suc_node.g = suc_node.g + current_node.g
                    suc_node.f = (h_weight*suc_node.h) + (1-h_weight)*suc_node.g
                    #suc_node.f = (h_weight * suc_node.h) + suc_node.g
                    # if the node has been closed move on
                    #it is possible to find a less expensive route to a node that's already been expanded
                    if suc_state in [s for s in closed.keys()]:
                        old_node_f = closed[suc_state].f
                        if suc_node.f < old_node_f:
                            closed.pop(suc_state)
                            opened[suc_state] = (suc_node.f, suc_state, suc_node)
                    # if the node hasn't been opened yet open it
                    if not suc_state in [s for s in opened] and not suc_state in closed:
                        opened[suc_state] = (suc_node.f, suc_state, suc_node)
                    # if the node has been opened but the new g is less than the g it had been originally inserted with, replace it
                    if suc_state in [s for s in opened.keys()]:
                        old_node_f = opened[suc_state][0]
                        if suc_node.f < old_node_f:
                            opened[suc_state] = (suc_node.f, suc_state, suc_node)

                    # IS-GOAL is checked in expansion time in wA*
                    # if self.env.is_final_state(suc_state):
                    #    return self.traverse_back_for_UCS_sol(suc_node, self.cost, [], 0)
            # in case of error
        return [0], -1.0, -1


class IDAStarAgent():
    def _init_(self):
        self.env = None
        self.expanded_num = 0
        self.new_limit = 0

    def manhattan_distance_for_multiple_goals(self, state_num, N):
        min = 100
        row, col = self.env.to_row_col(state_num)
        for goal in self.env.goals:
            g_row, g_col = self.env.to_row_col(goal)
            manhattan_distance = (g_row - row) + (g_col - col)
            if manhattan_distance <= min:
                min = manhattan_distance
        return min

    def h_sap_heuristic(self, state_num, N):
        # row , col = self.env.to_row_col(state_num)
        manhattan_distance = self.manhattan_distance_for_multiple_goals(state_num, N)
        cost_p = 100
        return min(manhattan_distance, cost_p)


    def traverse_back_for_IDA_sol(self, current_node, total_cost, actions_list, expanded) -> Tuple[
        List[int], float, int]:
        # if the  node doesn't have a father: it is an  initial state - stop
        if current_node.father is None:
            return actions_list, total_cost, expanded
        # change the state on the board
        self.env.set_state(current_node.state)
        # insert at the beginning of the actions list because we are travesing back
        actions_list.insert(0, current_node.action)
        total_cost = total_cost + current_node.cost
        return self.traverse_back_for_IDA_sol(current_node.father, total_cost, actions_list, expanded)

    def dfs_f (self, current_node, prev_f, f_limit, closed, opened):
        #str1 = str(current_node.state)
        #print("current state is ", str1, " f limit is ", str(f_limit))
        new_f = prev_f + current_node.h
        if new_f > f_limit:
            #print("going back")
            #print(str(f_limit))
            self.new_limit = min(new_f, self.new_limit)
            return [], -1.0, -1
        if self.env.is_final_state(current_node.state):
            return self.traverse_back_for_IDA_sol(current_node, 0, [], self.expanded_num)
        self.expanded_num = self.expanded_num + 1
        #closed.add(current_node.state)
        opened[current_node.state] = current_node
        #print("currently expanded ", str(self.expanded_num))
        if not None in self.env.succ(current_node.state)[1]:
            for suc_action, suc_state_cost_term in self.env.succ(current_node.state).items():
                # self.env.set_state(current_node.state)
                suc_state, suc_cost, suc_term = suc_state_cost_term
                suc_node = Node(suc_state, current_node, suc_action, suc_cost)
                # update new cost for the successors which is their own cost plus the cost of everything up until now which is basically g of the father node
                suc_node.h = self.h_sap_heuristic(suc_state, self.N)
                suc_node.g = suc_node.g + current_node.g
                suc_node.f = suc_node.h + suc_node.g
                #print("current succ is ", str(suc_node.state))
                #if suc_state in closed:
                #    continue
                if suc_state in opened.keys():
                    if opened[suc_state].f < suc_node.f:
                        continue
                result = self.dfs_f(suc_node,current_node.f, f_limit, closed, opened)
                if not result[1] == -1:
                    return result
        #print("no successors left")
        return [], -1.0, -1




    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        # create your own environment to be able to change and use it without altering the original
        self.env = env
        self.env.reset()
        # size of the board
        self.N = sqrt(self.env.observation_space.n)
        # define an initial state and create the first node in the graph
        initial_state = self.env.get_initial_state()
        initial_node = Node(initial_state, None, None, 0)
        initial_node.h = self.h_sap_heuristic(initial_state, self.N)
        initial_node.f = initial_node.h + initial_node.g
        # initial_node.f = (h_weight * initial_node.h) + initial_node.g
        # make sure the initial state is not a goal state
        if self.env.is_final_state(initial_state):
            return [0], 0.0, 0
        # reset the values for a new search
        self.expanded_num = 0
        self.new_limit = self.h_sap_heuristic(initial_state, self.N)

        while bool(1):
            opened = {}
            closed = set()
            f_limit = self.new_limit
            self.new_limit = float('inf')
            result = self.dfs_f(initial_node,0,f_limit, closed, opened)
            if not result[1] == -1:
                return result

        return [],-1.0, -1