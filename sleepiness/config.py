from scipy.spatial import distance as dist
from playsound import playsound

EYE_ASPECT_RATIO_THRESH = 0.25
FRAMES = 35
BAG_THRESHOLD = 60
HEAD_ANGLE_THRESHOLD = 15
MOUTH_ASPECT_RATIO_THRESHOLD = 1.6
INNER_MOUTH_ASPECT_RATIO_THRESHOLD = 0.3

sounds = {"E": "sleepiness/sounds/alarmEyes.mp3",
          "B": "sleepiness/sounds/alarmBags.mp3",
          "H": "sleepiness/sounds/alarmHead.mp3",
          "M": "sleepiness/sounds/alarmMouth.mp3",
          }

# these indices relate to the index convention of the landmark detection model
FACIAL_LANDMARKS_INDICES = {"mouth": (48, 68),
                            "inner_mouth": (60, 68),
                            "right_eyebrow": (17, 22),
                            "left_eyebrow": (22, 27),
                            "right_eye": (36, 42),
                            "left_eye": (42, 48),
                            "nose": (27, 36),
                            "jaw": (0, 17),
                            "left_eye_bottom": (36, 41, 40, 39),
                            "right_eye_bottom": (42, 47, 46, 45)}

FACIAL_LANDMARKS_COLORS = {"mouth": (255, 0, 0),
                           "inner_mouth": (0, 0, 255),
                           "right_eyebrow": (0, 0, 0),
                           "left_eyebrow": (0, 0, 0),
                           "right_eye": (0, 255, 0),
                           "left_eye": (0, 255, 0),
                           "eyes": (0, 255, 0),
                           "nose1": (0, 255, 255),
                           "nose2": (255, 255, 255),
                           "jaw": (255, 255, 255),
                           "left_eye_bottom": (0, 0, 0),
                           "right_eye_bottom": (0, 0, 0),
                           "eyes_bottom": (0, 0, 0)}


def audio_alert(path):
    playsound(path)


def aspect_ratio(part, point_coords):
    vertical_distances = []
    if part == 'eye':
        """
        By convention, the eye region is marked by 6 landmark points,
        like so:

            1     2
        0---|-----|---3
            5     4
        """

        # compute the euclidean distances between the two sets of
        # vertical eye landmarks. Meaning, distance between points 1-5 and 2-4
        vertical_distances.append(dist.euclidean(point_coords[1], point_coords[5]))
        vertical_distances.append(dist.euclidean(point_coords[2], point_coords[4]))

        # compute the euclidean distance between the set of
        # horizontal eye landmarks. Meaning, distance between points 0-3
        horizontal = dist.euclidean(point_coords[0], point_coords[3])
    elif part == 'mouth':
        # similar logic applies for the mouth landmarks
        for landmark_pair in ((1, 11), (2, 10), (3, 9), (4, 8), (5, 7)):
            vertical_distances.append(dist.euclidean(point_coords[landmark_pair[0]],
                                                     point_coords[landmark_pair[1]]))

        horizontal = dist.euclidean(point_coords[0], point_coords[6])
        ###
    elif part == 'inner_mouth':
        # similar logic applies for the mouth landmarks
        # for landmark_pair in ((2, 6)):
        #     vertical_distances.append(dist.euclidean(point_coords[landmark_pair[0]]))
        vertical_distances = dist.euclidean(point_coords[2], point_coords[6])
        horizontal = dist.euclidean(point_coords[0], point_coords[4])
        result = float(vertical_distances / horizontal)
        return result

        # compute the aspect ratio
    # formula comes from the paper: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    result = sum(vertical_distances) / (2 * horizontal)

    return result
