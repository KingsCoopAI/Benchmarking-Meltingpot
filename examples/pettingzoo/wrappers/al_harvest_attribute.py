import numpy as np
from scipy.spatial import KDTree
# from PIL import Image

background_place = [1, 1]
background_color = {'black': (40, 40, 40),
                    'white': (253, 253, 253),
                    'red': (200, 10, 10),
                    'green': (10, 200, 10),
                    'blue': (10, 10, 200)
                    }
background_color_list = list(background_color.keys())
background_color_pixel = list(background_color.values())
background_color_kdtree = KDTree(background_color_pixel)

agent_place = [6, 3]
agent_color = {'red': (200, 10, 10),
               'grey': (128, 128, 128),
               'blue': (10, 10, 200),
               'green': (10, 200, 10),
               'black': (55, 55, 55),
               'white': (255, 255, 255)
               }
agent_color_list = list(agent_color.keys())
agent_color_pixel = list(agent_color.values())
agent_color_kdtree = KDTree(agent_color_pixel)

tree_color = {
    'red': (200, 10, 10),
    'blue': (10, 10, 200),
    'green': (10, 200, 10),
    'grey': (55, 55, 55),
    'black': (0, 0, 0)
}
tree_color_list = list(tree_color.keys())
tree_color_pixel = list(tree_color.values())
tree_color_kdtree = KDTree(tree_color_pixel)

left_ear_place = [1, 2]
right_ear_place = [1, 5]
mouth_place = [4, 3]
tree_place = [3, 6]
background_tree_place_unhavest = [4, 2]
background_tree_place = [2, 3]
ear_color = {'black': np.array([[0, 0, 0]])}
mouth_color = {'white': np.array([[255, 255, 255]]),
               'grey': np.array([[128, 128, 128]])}


def sprite2attr_ah(points):
    bg_nearest = background_color_kdtree.query(points[:, background_place[0], background_place[1], :], k=1, p=1)[1]
    ag_nearest = agent_color_kdtree.query(points[:, agent_place[0], agent_place[1], :], k=1, p=1)[1]
    tree_nearest = tree_color_kdtree.query(points[:, tree_place[0], tree_place[1], :], k=1, p=1)[1]
    background_tree_nearest = tree_color_kdtree.query(points[:, background_tree_place[0],
                                                      background_tree_place[1], :], k=1, p=1)[1]
    background_tree_havest_nearest = tree_color_kdtree.query(points[:, background_tree_place_unhavest[0],
                                                             background_tree_place_unhavest[1], :], k=1, p=1)[1]
    left_ear_cond = np.sum(np.abs(points[:, left_ear_place[0], left_ear_place[1], :] - ear_color['black']),
                           axis=1) > 200
    right_ear_cond = np.sum(np.abs(points[:, right_ear_place[0], right_ear_place[1], :] - ear_color['black']),
                            axis=1) > 200
    mouth_cond1 = np.sum(np.abs(points[:, mouth_place[0], mouth_place[1], :] - mouth_color['white']), axis=1) < 10
    mouth_cond2 = np.sum(np.abs(points[:, mouth_place[0], mouth_place[1], :] - mouth_color['grey']), axis=1) < 10
    attr_list = []
    for i in range(points.shape[0]):
        bg, ag, tree, if_harvest, direction = (background_color_list[bg_nearest[i]], None, None, None, None)
        if bg == 'black':
            ag = agent_color_list[ag_nearest[i]]
            if ag != 'black':
                left_ear = left_ear_cond[i]
                right_ear = right_ear_cond[i]
                if left_ear and not right_ear:
                    direction = 'right'
                elif not left_ear and right_ear:
                    direction = 'left'
                elif right_ear and left_ear:
                    mouth = mouth_cond1[i] or mouth_cond2[i]
                    if mouth:
                        direction = 'front'
                    else:
                        direction = 'back'
                if tree_color_list[tree_nearest[i]] != 'grey':
                    tree = tree_color_list[tree_nearest[i]]
            else:
                tre = tree_color_list[background_tree_nearest[i]]
                if tre != 'grey' and tre != 'black':
                    tree = tre
                    harvest_color = tree_color_list[background_tree_havest_nearest[i]]
                    if harvest_color != 'grey' and harvest_color != 'black':
                        if_harvest = 'no'
                    else:
                        if_harvest = 'yes'
        attr_list.append({'Background': bg,
                          'Agent': ag,
                          'Tree': tree,
                          'Dir': direction,
                          'Harvest': if_harvest
                          })
    return attr_list