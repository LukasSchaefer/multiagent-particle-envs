import numpy as np
from multiagent.scenarios.simple_tag import Scenario
from multiagent.scenarios.scenario_util import obscure_pos, obscure_vel


class POScenario(Scenario):
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(obscure_pos(agent.state.p_pos, entity.state.p_pos))

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(obscure_pos(agent.state.p_pos, other.state.p_pos))
            if not other.adversary:
                other_vel.append(obscure_vel(agent.state.p_pos, other.state.p_pos, other.state.p_vel))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
