import numpy as np
from multiagent.scenarios.simple_adversary import Scenario
from multiagent.scenarios.scenario_util import obscure_pos, obscure_col


class POScenario(Scenario):
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(obscure_pos(agent.state.p_pos, entity.state.p_pos))

        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(obscure_col(agent.state.p_pos, entity.state.p_pos, entity.color))

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(obscure_pos(agent.state.p_pos, other.state.p_pos))

        if not agent.adversary:
            return np.concatenate([obscure_pos(agent.state.p_pos, agent.goal_a.state.p_pos)] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)
