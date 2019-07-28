import numpy as np

VIEW_DISTANCE = 1.5

def distance(p1, p2):
    vector = p2 - p1
    d = np.sqrt(np.sum(np.square(vector)))
    return vector, d

def obscure_pos(agent_pos, entity_pos):
    v, d = distance(agent_pos, entity_pos)
    if d <= VIEW_DISTANCE:
        return v
    else:
        return np.zeros(v.shape)

def obscure_vel(agent_pos, entity_pos, entity_vel):
    _, d = distance(agent_pos, entity_pos)
    if d <= VIEW_DISTANCE:
        return entity_vel
    else:
        return np.zeros(entity_vel.shape)

def obscure_col(agent_pos, entity_pos, entity_color):
    _, d = distance(agent_pos, entity_pos)
    if d <= VIEW_DISTANCE:
        return entity_color
    else:
        return np.zeros(entity_color.shape)
