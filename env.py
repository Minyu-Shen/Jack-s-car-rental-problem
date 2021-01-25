import math
import numpy as np
import copy
import parameters as paras

################################################################

poisson_cache = dict()


def poisson(n, lam):
    global poisson_cache
    key = n * 20 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = math.exp(-lam) * math.pow(lam, n) / math.factorial(n)
    return poisson_cache[key]


# print(poisson(5, 3))


class Env(object):
    def __init__(self) -> None:
        super().__init__()

    def init_state(self):
        self.state = [10, 10]

    def move_one_day(self, action):
        # state = copy.deepcopy(self.state)

        # first get action
        self.state[0] = min(self.state[0] - action, paras.MAX_CARS)
        self.state[1] = min(self.state[1] + action, paras.MAX_CARS)

        # then listen to the env
        req_1 = np.random.poisson(paras.REQ_MEAN_1, 1)
        req_2 = np.random.poisson(paras.REQ_MEAN_2, 1)
        ret_1 = np.random.poisson(paras.RET_MEAN_1, 1)
        ret_2 = np.random.poisson(paras.RET_MEAN_2, 1)
        # requests
        real_req_num_1 = min(self.state[0], req_1)
        real_req_num_2 = min(self.state[1], req_2)
        self.state[0] -= real_req_num_1
        self.state[1] -= real_req_num_2
        # returns
        self.state[0] = min(self.state[0] + ret_1, paras.MAX_CARS)
        self.state[1] = min(self.state[1] + ret_2, paras.MAX_CARS)
        # reward
        reward = (real_req_num_1 + real_req_num_2) * paras.RENT_REWARD
        # return the next_state
        return self.state

    def get_bellman_q(self, s, a, value, gamma):
        # for model-free, it is unknown
        # the complexity is O(n^4)
        # s : 2-d tuple, indicating no. of cars at place 1 and 2
        # a : scalar, posive means moving 1->2, negative means 2->1
        # value function: (MAX_CARS+1) * (MAX_CARS+1) matrix

        # because reward has no randomness, we don't "average" them
        expected_return = paras.MOVE_COST * abs(a)
        num_of_cars_1 = min(s[0] - a, paras.MAX_CARS)
        num_of_cars_2 = min(s[1] + a, paras.MAX_CARS)

        for req_1 in range(0, paras.TRUNCATE):
            for req_2 in range(0, paras.TRUNCATE):
                # real request number should not be greater than the current
                real_req_num_1 = min(num_of_cars_1, req_1)
                real_req_num_2 = min(num_of_cars_2, req_2)
                num_of_cars_1 -= real_req_num_1
                num_of_cars_2 -= real_req_num_2

                req_joint_prob = poisson(req_1, paras.REQ_MEAN_1) * poisson(
                    req_2, paras.REQ_MEAN_2
                )

                # reward is based on requested cars
                reward = (real_req_num_1 + real_req_num_2) * paras.RENT_REWARD

                # some cars will be returned
                for ret_1 in range(0, paras.TRUNCATE):
                    for ret_2 in range(0, paras.TRUNCATE):
                        num_of_cars_1_ = min(num_of_cars_1 + ret_1, paras.MAX_CARS)
                        num_of_cars_2_ = min(num_of_cars_2 + ret_2, paras.MAX_CARS)
                        ret_joint_prob = poisson(ret_1, paras.RET_MEAN_1) * poisson(
                            ret_2, paras.RET_MEAN_2
                        )
                        joint_prob = req_joint_prob * ret_joint_prob
                        expected_return += joint_prob * (
                            reward + gamma * value[num_of_cars_1_, num_of_cars_2_]
                        )
        return expected_return
