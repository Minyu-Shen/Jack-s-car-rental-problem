import parameters as paras
import numpy as np
import itertools
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt


class Sync_Policy_Iteration_Agent(object):
    def __init__(self, get_bellman_q) -> None:
        super().__init__()
        self.actions = np.arange(-paras.MAX_MOVE, paras.MAX_MOVE + 1)
        self.action_indexs = {el: ind[0] for ind, el in np.ndenumerate(self.actions)}
        self.policy = np.zeros((paras.MAX_CARS + 1, paras.MAX_CARS + 1), dtype=np.int)
        self.value = np.zeros(self.policy.shape)
        self.get_bellman_q = get_bellman_q

    def plot(self):
        print(self.policy)
        # plt.figure()
        # plt.xlim(0, paras.MAX_CARS + 1)
        # plt.ylim(0, paras.MAX_CARS + 1)
        # plt.table(cellText=np.flipud(self.policy), loc=(0, 0), cellLoc="center")
        # plt.show()

    def solve(self):
        for i in range(1):
            self.policy_evaluation()
            policy_change = self.policy_improvement()
            if policy_change == 0:
                break
        self.plot()

    def policy_improvement(self):
        # we now only have the v value function, we need q action-value function to act
        # so we estimate it by the bellman equation again
        new_policy = np.copy(self.policy)

        # q(s,a)
        q = np.zeros((paras.MAX_CARS + 1, paras.MAX_CARS + 1, np.size(self.actions)))
        action_cooks = {}
        with mp.Pool(processes=paras.NP) as p:
            for action in self.actions:  # for each a, update q(s, a)
                action_cooks[action] = partial(
                    self.expected_return_for_improve_closre, action
                )
                all_state_generator = (
                    (i, j)
                    for i, j in itertools.product(
                        np.arange(paras.MAX_CARS), np.arange(paras.MAX_CARS)
                    )
                )
                q_updates = p.map(action_cooks[action], all_state_generator)
                for s, a, q_value in q_updates:
                    q[s[0], s[1], self.action_indexs[a]] = q_value

        for s1 in range(q.shape[0]):
            for s2 in range(q.shape[1]):
                new_policy[s1, s2] = self.actions[np.argmax(q[s1, s2])]

        policy_change = (new_policy != self.policy).sum()
        print(f"Policy changed in {policy_change} states")
        self.policy = new_policy
        return policy_change

    def policy_evaluation(self):
        while True:
            new_value = np.copy(self.value)
            # sweep all the states, get the value updated
            all_state_generator = (
                (i, j)
                for i, j in itertools.product(
                    np.arange(paras.MAX_CARS), np.arange(paras.MAX_CARS)
                )
            )

            v_updates = []
            with mp.Pool(processes=paras.NP) as p:
                v_updates = p.map(
                    self.expected_return_for_eval_closure, all_state_generator
                )

            for s, v in v_updates:
                new_value[s[0], s[1]] = v
            v_diff = np.abs(new_value - self.value).sum()
            print("value difference in policy evaluation is:", v_diff)
            self.value = new_value

            if v_diff < paras.V_CONVERG:
                print("evaluation converges")
                break

    # the sweeping(updating) can be parallelized, w.r.t. each state
    def expected_return_for_eval_closure(self, state):
        action = self.policy[state[0], state[1]]  # pi(a|s)
        # because q_pi(s, pi(a|s)) = v_pi(s)
        new_v = self.get_bellman_q(state, action, self.value, paras.GAMMA)

        return state, new_v

    def expected_return_for_improve_closre(self, action, state):
        if (
            (action >= 0 and state[0] >= action)
            or (action < 0 and state[1] >= abs(action))
        ) == False:
            return state, action, -float("inf")
        q = self.get_bellman_q(state, action, self.value, paras.GAMMA)

        return state, action, q

