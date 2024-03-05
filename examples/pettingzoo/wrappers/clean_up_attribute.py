import numpy as np
# from PIL import Image
from scipy.spatial import KDTree


def f(tablet):
    if tablet == 'river1' or tablet == 'river2':
        return 'river'
    elif tablet == 'beam1' or tablet == 'beam3':
        return 'beamed sand'
    elif tablet == 'sand1' or tablet == 'sand2':
        return 'sand'
    elif tablet == 'grass1' or tablet == 'grass2':
        return 'grass'
    elif tablet == 'beamed grass1' or tablet == 'beamed grass2':
        return 'beamed grass'
    return tablet


background_place = [7, 7]

background_color = {
    'river1': (34, 132, 166),
    'river2': (27, 103, 129),
    'polluted water': (28, 152, 149),
    'beamed grass1': (119, 212, 190),
    'beam1': (137, 221, 224),
    'beamed river': (79, 194, 217),
    'beamed grass2': (108, 199, 185),
    'beam3': (122, 207, 212),
    'sand1': (219, 218, 186),
    'sand2': (174, 173, 148),
    'grass1': (164, 189, 75),
    'grass2': (129, 148, 59),
    'zapping beam': (252, 252, 106),
    'wall': (95, 95, 95)
}
background_color_list = list(background_color.keys())
background_color_pixel = list(background_color.values())
background_color_kdtree = KDTree(background_color_pixel)

mouth_place = [4, 3]
mouth_color = {
    'Cyan': (35, 185, 175),
    'Purple': (125, 50, 200),
    'Green': (125, 185, 65),
    'Pink': (205, 5, 165),
    'Red': (245, 65, 65),
    'Orange': (245, 130, 0),
    'Blue': (45, 110, 220),
    'Yellow': (195, 180, 0),
    'white': (255, 255, 255),
    'river1': (34, 132, 166),
    'river2': (27, 103, 129),
    'polluted water': (28, 152, 149),
    'beamed grass': (119, 212, 190),
    'beam1': (137, 221, 224),
    'beam2': (79, 194, 217),
    'beam3': (108, 199, 185),
    'beam4': (122, 207, 212),
    'sand1': (219, 218, 186),
    'sand2': (174, 173, 148),
    'grass1': (164, 189, 75),
    'grass2': (129, 148, 59),
}
mouth_color_list = list(mouth_color.keys())
mouth_color_pixel = list(mouth_color.values())
mouth_color_kdtree = KDTree(mouth_color_pixel)

agent_place = [2, 3]
agent_color = {
    'Cyan': (35, 185, 175),
    'Purple': (125, 50, 200),
    'Green': (125, 185, 65),
    'Pink': (205, 5, 165),
    'Red': (245, 65, 65),
    'Orange': (245, 130, 0),
    'Blue': (45, 110, 220),
    'Yellow': (195, 180, 0)
}
agent_color_list = list(agent_color.keys())
agent_color_pixel = list(agent_color.values())
agent_color_kdtree = KDTree(agent_color_pixel)

left_eye_place = [3, 2]
right_eye_place = [3, 5]
eye_color = {'eye': np.array([[60, 60, 60]])}
eye_color_beam = {'eye': np.array([[87, 172, 185]])}

mouth_color_beam = {
    'White': (148, 233, 246),
    'Green': (107, 211, 186),
    'Blue': (82, 188, 235),
    'Orange': (145, 194, 166),
    'Yellow': (129, 210, 166),
    'Pink': (132, 155, 218),
    'Purple': (107, 169, 229),
    'Red': (145, 173, 186),
    'Cyan': (79, 211, 221),
    'beamed grass': (119, 212, 190),
    'beam1': (137, 221, 224),
    'beam2': (79, 194, 217),
    'beam3': (108, 199, 185),
    'beam4': (122, 207, 212),
}
mouth_color_beam_list = list(mouth_color_beam.keys())
mouth_color_beam_pixel = list(mouth_color_beam.values())
mouth_color_beam_kdtree = KDTree(mouth_color_beam_pixel)

agent_color_beam = {
    'Green': (107, 211, 186),
    'Blue': (82, 188, 235),
    'Orange': (145, 194, 166),
    'Yellow': (129, 210, 166),
    'Pink': (132, 155, 218),
    'Purple': (107, 169, 229),
    'Red': (145, 173, 186),
    'Cyan': (79, 211, 221),
}
agent_color_beam_list = list(agent_color_beam.keys())
agent_color_beam_pixel = list(agent_color_beam.values())
agent_color_beam_kdtree = KDTree(agent_color_beam_pixel)

apple_place = [4, 2]
apple_color_beam = {'apple': np.array([[134, 178, 184]])}
apple_color = {'apple': np.array([[212, 80, 57]])}



def sprite2attr_cu(points):
    attr_list = []
    bg_nearest = background_color_kdtree.query(points[:, background_place[0], background_place[1], :], k=1)[1]
    ag_nearest = agent_color_kdtree.query(points[:, agent_place[0], agent_place[1], :], k=1)[1]
    ag_beam_nearest = agent_color_beam_kdtree.query(points[:, agent_place[0], agent_place[1], :], k=1)[1]
    mouth_nearest = mouth_color_kdtree.query(points[:, mouth_place[0], mouth_place[1], :], k=1)[1]
    mouth_beam_nearest = mouth_color_beam_kdtree.query(points[:, mouth_place[0], mouth_place[1], :], k=1)[1]
    apple = np.sum(np.abs(points[:, apple_place[0], apple_place[1], :] - apple_color['apple']), axis=1) < 10
    apple_beam = np.sum(np.abs(points[:, apple_place[0], apple_place[1], :] - apple_color_beam['apple']), axis=1) < 10
    left_eye_cond = np.sum(np.abs(points[:, left_eye_place[0], left_eye_place[1], :] - eye_color['eye']), axis=1) < 10
    right_eye_cond = np.sum(np.abs(points[:, right_eye_place[0], right_eye_place[1], :] - eye_color['eye']),
                            axis=1) < 10
    left_eye_beam_cond = np.sum(np.abs(points[:, left_eye_place[0], left_eye_place[1], :] - eye_color_beam['eye']),
                                axis=1) < 10
    right_eye_beam_cond = np.sum(np.abs(points[:, right_eye_place[0], right_eye_place[1], :] - eye_color_beam['eye']),
                                 axis=1) < 10

    for i in range(points.shape[0]):
        bg, mouth, ag, direction = background_color_list[bg_nearest[i]], None, None, None
        bg = f(bg)
        if bg == 'river' or bg == 'sand' or bg == 'polluted water':
            mouth = f(mouth_color_list[mouth_nearest[i]])
            if bg == mouth:
                ag = None
            else:
                ag = agent_color_list[ag_nearest[i]] + ' agent'
                if mouth == 'white':
                    if left_eye_cond[i]:
                        direction = 'front' if right_eye_cond[i] else 'left'
                    else:
                        direction = 'right' if right_eye_cond[i] else 'back'
                else:
                    direction = 'back'
        elif bg == 'beam':
            mouth = f(mouth_color_beam_list[mouth_beam_nearest[i]])
            if bg == mouth:
                ag = None
            else:
                ag = agent_color_beam_list[ag_beam_nearest[i]] + ' agent in cleaning beam'
                if mouth == 'white':
                    if left_eye_beam_cond[i]:
                        direction = 'front' if right_eye_beam_cond[i] else 'left'
                    else:
                        direction = 'right' if right_eye_beam_cond[i] else 'back'
                else:
                    direction = 'back'
        elif bg == 'beamed grass':
            if apple_beam[i]:
                ag = 'apple in cleaning beam'
            else:
                mouth = f(mouth_color_beam_list[mouth_beam_nearest[i]])
                if bg == mouth:
                    ag = None
                else:
                    ag = agent_color_beam_list[ag_beam_nearest[i]] + ' agent in cleaning beam'
                    if mouth == 'white':
                        if left_eye_beam_cond[i]:
                            direction = 'front' if right_eye_beam_cond[i] else 'left'
                        else:
                            direction = 'right' if right_eye_beam_cond[i] else 'back'
                    else:
                        direction = 'back'
        elif bg == 'grass':
            if apple[i]:
                ag = 'apple'
            else:
                mouth = f(mouth_color_list[mouth_nearest[i]])
                if bg == mouth:
                    ag = None
                else:
                    ag = agent_color_list[ag_nearest[i]] + ' agent'
                    if mouth == 'white':
                        if left_eye_cond[i]:
                            direction = 'front' if right_eye_cond[i] else 'left'
                        else:
                            direction = 'right' if right_eye_cond[i] else 'back'
                    else:
                        direction = 'back'
        attr_list.append({'Background': bg,
                          'Agent/object': ag,
                          'Agent dir': direction
                          })
    return attr_list