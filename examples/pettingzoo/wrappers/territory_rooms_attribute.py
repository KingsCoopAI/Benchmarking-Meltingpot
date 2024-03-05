import numpy as np
# from PIL import Image
from scipy.spatial import KDTree

background_place = [6, 1]
background_color = {
                    'black': (0, 0, 0),
                    'gray': (61, 61, 61),
                    'solid gray': (61, 57, 55),
                    'active yellow': (132, 127, 13),
                    'yellow': (100, 96, 43),
                    'active red': (155, 59, 58),
                    'red': (115, 62, 62),
                    'active blue':(37,85,150),
                    'blue': (56,75,108),
                    'active orange':(155,97,20),
                    'orange': (115,81,43),
                    'active pink':(131,23,117),
                    'pink': (103,45,92),
                    'active cyan':(32,124,126),
                    'cyan': (53,97,95),
                    'active dark purple':(86,42,131),
                    'dark purple': (80,58,102),
                    'active green': (90,129,51),
                    'green': (80,97,62),
                    'active purple':(114, 19, 140),
                    'purple': (90, 47, 102),
                    'zapping beam':(252, 252, 106)
                    }
background_color_list = list(background_color.keys())
background_color_pixel = list(background_color.values())
background_color_kdtree = KDTree(background_color_pixel)

agent_place = [6, 3]
agent_color = {
                'black': (0, 0, 0),
                'yellow': (195, 180, 0),
                'red': (245, 65, 65),
                'purple': (160, 15, 200),
                'blue': (45, 110, 220),
                'orange': (245, 130, 0),
                'pink': (205, 5, 165),
                'cyan': (35, 185, 175),
                'dark purple': (125, 50, 200),
                'green': (125, 185, 65)
               }
agent_color_list = list(agent_color.keys())
agent_color_pixel = list(agent_color.values())
agent_color_kdtree = KDTree(agent_color_pixel)

hand_color = {
                'black' : (0, 0, 0),
                'yellow': (146, 135, 0),
                'red': (183, 48, 48),
                'purple': (120, 11, 150),
                'blue': (33, 82, 165),
                'orange': (183, 97, 0),
                'pink': (153, 3, 123),
                'cyan': (26, 138, 131),
                'dark purple': (93, 37, 150),
                'green': (93, 138, 48),
                'brown': (143, 96, 74)
            }
hand_color_list = list(hand_color.keys())
hand_color_pixel = list(hand_color.values())
hand_color_kdtree = KDTree(hand_color_pixel)

paddle_place = {
                'Left': [[3,2], [4,2], [5,2], [3,4]],
                'Right': [[3,5], [4,5], [5,5], [3,2]],
                'Down': [[7,5], [7,6], [6,6], [6,5]],
                'Up': [[0,5], [0,6], [0,7], [2,7]]
              }
paddle_color={
    'paddle':np.array([[199,176,135]]),
    'brown':np.array([[70,70,70]])
    # 'Yellow':(195,180,0),
    # 'Cyan':(35,185,175),
    # 'Dark Purple':(160,15,200),
    # 'Orange':(245,130,0),
    # 'Green':(125,185,65),
    # 'Purple':(195,180,0),
    # 'Pink':(205,5,165),
    # 'Red':(245,65,65),
    # 'Blue':(45,110,220)
    }

light_gray_color = {'light_gray': np.array([[80, 80, 80]])}
light_gray_place = [5, 2]

left_ear_place = [1, 3]
right_ear_place = [1, 4]
left_hand_place = [4, 0]
right_hand_place = [4, 7]
back_place = [[3, 6], [5, 6]]
broken_color = {'black': np.array([[0, 0, 0]])}
broken_place = [3, 3]


def sprite2attr_tr(points):
    attr_list = []
    bg_nearest = background_color_kdtree.query(points[:, background_place[0], background_place[1], :], k=1, p=1)[1]
    ag_nearest = agent_color_kdtree.query(points[:, agent_place[0], agent_place[1], :], k=1, p=1)[1]
    broken_con = np.sum(np.abs(points[:, broken_place[0], broken_place[1], :] - broken_color['black']), axis=1) < 10
    resource_con = np.sum(np.abs(points[:, light_gray_place[0], light_gray_place[1], :] - light_gray_color['light_gray']), axis=1) < 20
    left_ear_nearest = agent_color_kdtree.query(points[:, left_ear_place[0], left_ear_place[1], :], k=1, p=1)[1]
    right_ear_nearest = agent_color_kdtree.query(points[:, right_ear_place[0], right_ear_place[1], :], k=1, p=1)[1]
    left_hand_nearest = hand_color_kdtree.query(points[:, left_hand_place[0], left_hand_place[1], :], k=1, p=1)[1]
    right_hand_nearest = hand_color_kdtree.query(points[:, right_hand_place[0], right_hand_place[1], :], k=1, p=1)[1]
    back_nearest1 = hand_color_kdtree.query(points[:, back_place[0][0], back_place[0][1], :], k=1, p=1)[1]
    back_nearest2 = hand_color_kdtree.query(points[:, back_place[1][0], back_place[1][1], :], k=1, p=1)[1]
    down_flag1 = np.sum(
        np.abs(points[:, paddle_place['Down'][0][0], paddle_place['Down'][0][1], :] - paddle_color['paddle']),
        axis=1) < 10
    down_flag2 = np.sum(
        np.abs(points[:, paddle_place['Down'][1][0], paddle_place['Down'][1][1], :] - paddle_color['paddle']),
        axis=1) < 10
    down_flag3 = np.sum(
        np.abs(points[:, paddle_place['Down'][2][0], paddle_place['Down'][2][1], :] - paddle_color['paddle']),
        axis=1) < 10
    up_flag1 = np.sum(np.abs(points[:, paddle_place['Up'][0][0], paddle_place['Up'][0][1], :] - paddle_color['brown']),
                      axis=1) < 10
    up_flag2 = np.sum(np.abs(points[:, paddle_place['Up'][1][0], paddle_place['Up'][1][1], :] - paddle_color['brown']),
                      axis=1) < 10
    up_flag3 = np.sum(np.abs(points[:, paddle_place['Up'][2][0], paddle_place['Up'][2][1], :] - paddle_color['brown']),
                      axis=1) < 10
    left_flag1 = np.sum(
        np.abs(points[:, paddle_place['Left'][0][0], paddle_place['Left'][0][1], :] - paddle_color['brown']),
        axis=1) < 10
    left_flag2 = np.sum(
        np.abs(points[:, paddle_place['Left'][1][0], paddle_place['Left'][1][1], :] - paddle_color['brown']),
        axis=1) < 10
    left_flag3 = np.sum(
        np.abs(points[:, paddle_place['Left'][2][0], paddle_place['Left'][2][1], :] - paddle_color['brown']),
        axis=1) < 10
    right_flag1 = np.sum(
        np.abs(points[:, paddle_place['Right'][0][0], paddle_place['Right'][0][1], :] - paddle_color['brown']),
        axis=1) < 10
    right_flag2 = np.sum(
        np.abs(points[:, paddle_place['Right'][1][0], paddle_place['Right'][1][1], :] - paddle_color['brown']),
        axis=1) < 10
    right_flag3 = np.sum(
        np.abs(points[:, paddle_place['Right'][2][0], paddle_place['Right'][2][1], :] - paddle_color['brown']),
        axis=1) < 10

    for i in range(points.shape[0]):
        ag, resource, direction = None, None, None
        bg = background_color_list[bg_nearest[i]]
        if bg == 'black':# agent
            ag = agent_color_list[ag_nearest[i]]
            left_ear_color = agent_color_list[left_ear_nearest[i]]
            right_ear_color = agent_color_list[right_ear_nearest[i]]
            left_hand_color = hand_color_list[left_hand_nearest[i]]
            right_hand_color = hand_color_list[right_hand_nearest[i]]
            back_color1 = hand_color_list[back_nearest1[i]]
            back_color2 = hand_color_list[back_nearest2[i]]
            if ag == 'black':
                ag = None
            else:
                left_ear = (left_ear_color == ag)
                right_ear = (right_ear_color == ag)
                left_hand = (left_hand_color == ag)
                right_hand = (right_hand_color == ag)
                back = (back_color1 == ag)
                if left_hand:
                    direction = 'left'
                elif right_hand:
                    direction = 'right'
                else:
                    if right_ear:
                        direction = 'right'
                    elif left_ear:
                        direction = 'left'
                    else:
                        if back:
                            direction = 'back'
                        elif back_color2 != ag:
                            direction = 'back'
                        else:
                            direction = 'front'
        else:#resource
            if bg == 'gray':
                if resource_con[i]:
                    resource = 'broken ' + bg if broken_con[i] else bg
                else:
                    resource = 'wall'
            else:
                resource = 'broken ' + bg if broken_con[i] else bg
        paddle_dir = None
        if down_flag1[i] and down_flag2[i] and down_flag3[i]:
            paddle_dir = 'down'
        elif up_flag1[i] and up_flag2[i] and up_flag3[i]:
            paddle_dir = 'up'
        elif left_flag1[i] and left_flag2[i] and left_flag3[i]:
            paddle_dir = 'left'
        elif right_flag1[i] and right_flag2[i] and right_flag3[i]:
            paddle_dir = 'right'
        attr_list.append({'Background': bg,
                          'Agent': ag,
                          'Resource': resource,
                          'Direction': direction,
                          'Paddle dir': paddle_dir,
                          })
    return attr_list
