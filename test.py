import copy
import numpy as np


class A:
    def __init__(self) -> None:
        self.prop = [5, 6]

    def method(self):
        prop = copy.deepcopy(self.prop)
        self.prop[0] = 100
        print(prop)
        print(self.prop[0])


# a = A()
# print(a.prop)
# a.method()

# actions = np.arange(-5, 6)
# inverse_actions = {el: ind[0] for ind, el in np.ndenumerate(actions)}
# print(actions)
# print(inverse_actions)


q = np.array(list(range(12))).reshape(2, 2, 3)

print(q[1, 1])

