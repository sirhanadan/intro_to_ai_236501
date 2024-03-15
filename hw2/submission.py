import time
import heapdict

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, Robot, Package
import random


# TODO: section a : 3


def smart_heuristic(env: WarehouseEnv, agent_id: int):
    robot = env.get_robot(agent_id)
    opponent_id = 0
    if agent_id == 0:
        opponent_id = 1
    opponent = env.get_robot(opponent_id)
    if env.done():
        if robot.credit > opponent.credit:
            return 9999999
        if robot.credit < opponent.credit:
            return -9999999
        return 0
    else:
        if robot.package is None: # if packages not on board
            packages = [env.packages[0], env.packages[1]]
            distances = heapdict.heapdict()
            for p in packages:
                package_agent_distance = manhattan_distance(robot.position, p.position)
                distances[p.position] = package_agent_distance
            closest_package = distances.popitem()
            return robot.credit*100 - closest_package[1] # credit - distance to package
        else:
            p = robot.package # returns list?
            agent_destination_dist = manhattan_distance(robot.position, p.destination)
            package_destination_dist = manhattan_distance(p.position, p.destination)
            return robot.credit*100 - agent_destination_dist + package_destination_dist + 25
    return 0



class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)




class AgentMinimax(Agent):
    def __init__(self):
        self.epsilon = 0.98
        self.kill_time = None
        self.agent = None


    def get_operators(self, env: WarehouseEnv, whose_turn):
        return env.get_legal_operators(whose_turn)

    def get_op_succ_tup(self, env: WarehouseEnv, whose_turn):
        operators = self.get_operators(env, whose_turn)
        list_of_op_succ_tups = []
        for op in operators:
            cloned_env = env.clone()
            cloned_env.apply_operator(whose_turn, op)
            tup = (op, cloned_env)
            list_of_op_succ_tups.append(tup)
        for tup in list_of_op_succ_tups:
            if tup[1] is None:
                raise NotImplementedError()

        return list_of_op_succ_tups

    def termination_condition(self, env: WarehouseEnv, depth):
        #stop the iteration when the time is over, the depth is 0, or the game is over
        if time.time() > self.kill_time:
            return True
        if depth == 0:
            return True
        if env.done():#done is w\ battaries and credits
            return True
        return False

    def next_turn(self, cur_turn):# whose turn is next, between the two agents
        if cur_turn == 0:
            return 1
        else:
            return 0

    def minimax_iteration(self, env: WarehouseEnv, whose_turn, depth):
        #elimimation condition
        if self.termination_condition(env, depth):
            return smart_heuristic(env, whose_turn) #return utility !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #a list of tuples of operators and the successor environments we get from applying these operators
        op_suc = self.get_op_succ_tup(env, whose_turn)

        # the current player is our agent, he is a max player+
        if whose_turn == self.agent:
            curMax = float('-inf')
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return smart_heuristic(cloned_env, whose_turn)

                next_turn = self.next_turn(whose_turn)

                v = self.minimax_iteration(cloned_env, next_turn, depth - 1)

                if v > curMax:
                    curMax = v

            return curMax

        else: # if the player currently playing is not our agent, he is a minimum player (rival)
            curMin = float('inf')
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return smart_heuristic(cloned_env, whose_turn)

                next_turn = self.next_turn(whose_turn)

                v = self.minimax_iteration(cloned_env, next_turn, depth - 1)

                if v < curMin:
                    curMin = v

            return curMin

        return 0 #should never reach this



    def minimax_anytime(self, env: WarehouseEnv, agent):
        cur_max = float('-inf')
        depth = 1
        envi = env.clone()
        ops = envi.get_legal_operators(agent)
        operator = None
        #if the list of operators (ops) isn't empty, update 'operator' to be the first legal op
        operator = "move north"

        while (time.time() <= self.kill_time):
            local_max = float('-inf')
            op_suc = self.get_op_succ_tup(envi, agent)
            local_op = op_suc[0][0]
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return operator
                v = self.minimax_iteration(cloned_env, agent, depth)
                if v >= local_max:
                    local_max = v
                    local_op = op
            if local_max >= cur_max:
                cur_max = local_max
                operator = local_op

            depth = depth + 1

        return operator



    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent = agent_id
        self.kill_time = time.time() + (self.epsilon) * time_limit
        return self.minimax_anytime(env, agent_id)

        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def __init__(self):
        self.epsilon = 0.9
        self.kill_time = None
        self.agent = None



    def get_operators(self, env: WarehouseEnv, whose_turn):
        return env.get_legal_operators(whose_turn)

    def get_op_succ_tup(self, env: WarehouseEnv, whose_turn):
        operators = self.get_operators(env, whose_turn)
        list_of_op_succ_tups = []
        for op in operators:
            cloned_env = env.clone()
            cloned_env.apply_operator(whose_turn, op)
            tup = (op, cloned_env)
            list_of_op_succ_tups.append(tup)
        for tup in list_of_op_succ_tups:
            if tup[1] is None:
                raise NotImplementedError()

        return list_of_op_succ_tups

    def termination_condition(self, env: WarehouseEnv, depth):
        #stop the iteration when the time is over, the depth is 0, or the game is over
        if time.time() > self.kill_time:
            return True
        if depth == 0:
            return True
        if env.done():#done is w\ battaries and credits
            return True
        return False

    def next_turn(self, cur_turn):# whose turn is next, between the two agents
        if cur_turn == 0:
            return 1
        else:
            return 0

    def alpha_beta_iteration(self, env: WarehouseEnv, whose_turn, depth, a, b):
        #elimimation condition
        if self.termination_condition(env, depth):
            return smart_heuristic(env, whose_turn)

        #a list of tuples of operators and the successor environments we get from applying these operators
        op_suc = self.get_op_succ_tup(env, whose_turn)

        # the current player is our agent, he is a max player+
        if whose_turn == self.agent:
            curMax = float('-inf')
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return smart_heuristic(cloned_env, whose_turn)

                next_turn = self.next_turn(whose_turn)

                v = self.alpha_beta_iteration(cloned_env, next_turn, depth - 1, a, b)

                if v > curMax:
                    curMax = v

                a = max(curMax, a)

                if curMax >= b:
                    return float('inf')

            return curMax

        else: # if the player currently playing is not our agent, he is a minimum player (rival)
            curMin = float('inf')
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return smart_heuristic(cloned_env, whose_turn)

                next_turn = self.next_turn(whose_turn)

                v = self.alpha_beta_iteration(cloned_env, next_turn, depth - 1, a, b)

                if v < curMin:
                    curMin = v

                b = min(curMin, b)

                if curMin <= a:
                    return float('-inf')

            return curMin

        return 0 #should never reach this



    def alpha_beta_anytime(self, env: WarehouseEnv, agent):
        cur_max = float('-inf')
        depth = 1
        envi = env.clone()
        ops = envi.get_legal_operators(agent)
        operator = None
        # if the list of operators (ops) isn't empty, update 'operator' to be the first legal op
        operator = "move north"

        while (time.time() <= self.kill_time):
            local_max = float('-inf')
            op_suc = self.get_op_succ_tup(envi, agent)
            local_op = op_suc[0][0]
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return operator
                v = self.alpha_beta_iteration(cloned_env, agent, depth, float('-inf'), float('inf'))
                if v >= local_max:
                    local_max = v
                    local_op = op
            if local_max >= cur_max:
                cur_max = local_max
                operator = local_op

            depth = depth + 1

        return operator

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent = agent_id
        self.kill_time = time.time() + (self.epsilon) * time_limit

        return self.alpha_beta_anytime(env.clone(), agent_id)


        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def __init__(self):
        self.epsilon = 0.98
        self.kill_time = None
        self.agent = None


    def get_operators(self, env: WarehouseEnv, whose_turn):
        return env.get_legal_operators(whose_turn)

    def get_op_succ_tup(self, env: WarehouseEnv, whose_turn):
        operators = self.get_operators(env, whose_turn)
        list_of_op_succ_tups = []
        for op in operators:
            cloned_env = env.clone()
            cloned_env.apply_operator(whose_turn, op)
            tup = (op, cloned_env)
            list_of_op_succ_tups.append(tup)
        for tup in list_of_op_succ_tups:
            if tup[1] is None:
                raise NotImplementedError()

        return list_of_op_succ_tups

    def termination_condition(self, env: WarehouseEnv, depth):
        #stop the iteration when the time is over, the depth is 0, or the game is over
        if time.time() > self.kill_time:
            return True
        if depth == 0:
            return True
        if env.done():#done is w\ battaries and credits
            return True
        return False

    def next_turn(self, cur_turn):# whose turn is next, between the two agents
        if cur_turn == 0:
            return 1
        else:
            return 0

    def propability(self, env: WarehouseEnv, whose_turn : int):
        # the probability for each move is 1/7
        # but if there exists a charging station right next to me (one of the neighboring squares) thr probability of moving towards it is 2/7
        robot = env.get_robot(whose_turn)
        legal_operators = env.get_legal_operators(whose_turn)
        moving_operator = ["move north", "move south", "move east", "move west"]
        l = len(legal_operators)
        prob = 1/l
        #if one of the operators is 'charge', return 2/l
        for op in moving_operator:
            if op in legal_operators:
                cloned_env = env.clone()
                cloned_env.apply_operator(whose_turn, op)
                if "charge" in cloned_env.get_legal_operators(whose_turn):
                    return 2*prob

        return prob


    def expictimax_minimax_iteration(self, env: WarehouseEnv, whose_turn, depth):
        #elimimation condition
        if self.termination_condition(env, depth):
            return smart_heuristic(env, whose_turn) #return utility !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #a list of tuples of operators and the successor environments we get from applying these operators
        op_suc = self.get_op_succ_tup(env, whose_turn)

        # the current player is our agent, he is a max player+
        if whose_turn == self.agent:
            curMax = float('-inf')
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return smart_heuristic(cloned_env, whose_turn)

                next_turn = self.next_turn(whose_turn)

                v = self.expictimax_minimax_iteration(cloned_env, next_turn, depth - 1)

                if v > curMax:
                    curMax = v

            return curMax

        else: # if the player currently playing is not our agent, he is a minimum player (rival)
            curMin = float('inf')
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return smart_heuristic(cloned_env, whose_turn)

                next_turn = self.next_turn(whose_turn)

                prob = self.propability(cloned_env, whose_turn)
                v = self.expictimax_minimax_iteration(cloned_env, next_turn, depth - 1)

                if (prob*v) < curMin:
                    curMin = (prob*v)

            return curMin

        return 0 #should never reach this



    def expictimax_minimax_anytime(self, env: WarehouseEnv, agent):
        cur_max = float('-inf')
        depth = 1
        envi = env.clone()
        ops = envi.get_legal_operators(agent)
        operator = None
        #if the list of operators (ops) isn't empty, update 'operator' to be the first legal op
        operator = "move north"

        while (time.time() <= self.kill_time):
            local_max = float('-inf')
            op_suc = self.get_op_succ_tup(envi, agent)
            local_op = op_suc[0][0]
            for op, cloned_env in op_suc:
                if time.time() > self.kill_time:
                    return operator
                v = self.expictimax_minimax_iteration(cloned_env, agent, depth)
                if v >= local_max:
                    local_max = v
                    local_op = op
            if local_max >= cur_max:
                cur_max = local_max
                operator = local_op

            depth = depth + 1

        return operator



    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent = agent_id
        self.kill_time = time.time() + (self.epsilon) * time_limit
        return self.expictimax_minimax_anytime(env, agent_id)

        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)