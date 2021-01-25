from agent import Sync_Policy_Iteration_Agent
from env import Env

env = Env()
sync_pi_agent = Sync_Policy_Iteration_Agent(env.get_bellman_q)
sync_pi_agent.solve()
