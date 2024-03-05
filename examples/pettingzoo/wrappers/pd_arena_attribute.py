import numpy as np
from scipy.spatial import KDTree
# from PIL import Image

background_place = [1, 1]
background_color = {'black':(0, 0, 0),
                    'wall': (95, 95, 95),
                    'yellow':(252, 252, 106),
                    'red':(225, 30, 70),
                    'green': (30, 225, 185),
                    }
background_color_list = list(background_color.keys())
background_color_pixel = list(background_color.values())
background_color_kdtree = KDTree(background_color_pixel)

agent_place = [6, 3]
agent_color = {'black': (0, 0, 0),
               'yellow':(255, 229, 2),
               'red': (255, 0, 86),
               'purple': (158, 0, 142),
               'blue': (1, 0, 103),
               'sky': (50, 100, 200),
               'light': (213, 255, 0),
               'ocean': (14, 76, 161),
               'green': (0, 255, 0),
               'deep_green': (0, 95, 57),
               }
agent_color_list = list(agent_color.keys())
agent_color_pixel = list(agent_color.values())
agent_color_kdtree = KDTree(agent_color_pixel)

hat_place = [1, 2]
hat_color = {'white' :(204, 203, 200),
              'red'   :(139, 0, 0),
              'black' :(0, 0, 0),
              'yellow':(253, 184, 1),
              'green' :(0, 102, 0),
              'blue'  :(2, 71, 254)
              }
hat_color_list = list(hat_color.keys())
hat_color_pixel = list(hat_color.values())
hat_color_kdtree = KDTree(hat_color_pixel)

if_have_hat = [[1, 3], [1, 4]]
left_eye_place = [3, 2]
right_eye_place = [3, 5]
eye_color = {'eye':np.array([[60, 60, 60]])}


def sprite2attr_pa(points):
    # points: 121*8*8*3
    bg_nearest = background_color_kdtree.query(points[:, background_place[0], background_place[1], :], k=1, p=1)[1]
    ag_nearest = agent_color_kdtree.query(points[:, agent_place[0], agent_place[1], :], k=1, p=1)[1]
    hat_nearest = hat_color_kdtree.query(points[:, hat_place[0], hat_place[1], :], k=1, p=1)[1]
    left_eye_cond = np.sum(np.abs(points[:, left_eye_place[0], left_eye_place[1], :] - eye_color['eye']), axis = 1) < 10
    right_eye_cond = np.sum(np.abs(points[:, right_eye_place[0], right_eye_place[1], :] - eye_color['eye']), axis = 1) < 10
    no_hat_cond_0 = np.sum(np.abs(points[:, if_have_hat[0][0], if_have_hat[0][1], :] - hat_color['black']), axis = 1) < 10
    no_hat_cond_1 = np.sum(np.abs(points[:, if_have_hat[1][0], if_have_hat[1][1], :] - hat_color['black']), axis = 1) < 10

    attr_list = []
    for i in range(points.shape[0]):
        bg, ag, direction, hat = background_color_list[bg_nearest[i]], None, None, None
        if bg == 'black':
            ag = agent_color_list[ag_nearest[i]]
            if ag != 'black':
                if left_eye_cond[i]:
                    direction = 'front' if right_eye_cond[i] else 'left'
                else:
                    direction = 'right' if right_eye_cond[i] else 'back'
                if_hat1 = no_hat_cond_0[i] or no_hat_cond_1[i]
                if not if_hat1:
                    hat = hat_color_list[hat_nearest[i]]
        attr_list.append(
            {'Background': bg,
            'Agent': ag,
            'Direction': direction,
            'Hat': hat
            })
    return attr_list
