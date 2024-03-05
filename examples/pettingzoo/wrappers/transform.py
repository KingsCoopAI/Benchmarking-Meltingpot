from wrappers.al_harvest_attribute import sprite2attr_ah
from wrappers.clean_up_attribute import sprite2attr_cu
from wrappers.pd_arena_attribute import sprite2attr_pa
from wrappers.territory_rooms_attribute import sprite2attr_tr
import numpy as np

def obs2attr(obs, substrate):
    if substrate == 'allelopathic_harvest__open':
        sprite2attr = sprite2attr_ah
    elif substrate == 'clean_up':
        sprite2attr = sprite2attr_cu
    elif substrate == 'prisoners_dilemma_in_the_matrix__arena':
        sprite2attr = sprite2attr_pa
    elif substrate == 'territory__rooms':
        sprite2attr = sprite2attr_tr

    blocks = np.array_split(obs, 11, axis=0)
    blocks = [np.array_split(block, 11, axis=1) for block in blocks]
    blocks = np.array(blocks).reshape((121, 8, 8, 3))
    return sprite2attr(blocks)
