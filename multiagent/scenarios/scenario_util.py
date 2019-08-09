import numpy as np
from multiagent.core import Landmark

# partial observability
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

# observation noise
OBS_NOISE_DIM = 5
OBS_NOISE_VAR = 5

def get_noise():
    return [np.random.randn(OBS_NOISE_DIM) * OBS_NOISE_VAR]

# environment noise
ENV_NOISE_VAR = 5
ENV_NOISE_DISTANCE = 0.5

def create_noise_field(world_dim):
    noise_field = Landmark()
    noise_field.name = 'noise field'
    noise_field.collide = False
    noise_field.movable = False
    noise_field.boundary = False
    noise_field.size = ENV_NOISE_DISTANCE
    noise_field.state.p_pos = np.random.uniform(-1, 1, world_dim)
    noise_field.state.p_vel = np.zeros(world_dim)
    noise_field.color = np.array([0.3,0.3,0.3])
    return noise_field

def noise_field(agent_pos, noise_pos, obs):
    _, d = distance(agent_pos, noise_pos)
    if d <= ENV_NOISE_DISTANCE:
        noise = np.random.standard_normal(obs.shape) * ENV_NOISE_VAR
        return obs + noise
    else:
        return obs
