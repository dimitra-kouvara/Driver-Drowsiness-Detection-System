from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
import config as cfg

img = cv2.imread('bags1.jpg', 0)

# initialize a face detector
detector = dlib.get_frontal_face_detector()

# initialize landmark predictor
predictor = dlib.shape_predictor('landmark_predictor.dat')

left_eye_start, left_eye_end = cfg.FACIAL_LANDMARKS_INDICES["left_eye"]
right_eye_start, right_eye_end = cfg.FACIAL_LANDMARKS_INDICES["right_eye"]

# detect faces
face_rectangles = detector(img, 0)

# loop over the face detections
for face_rectangle in face_rectangles:
    # determine the facial landmarks for the face region
    shape = predictor(img, face_rectangle)

    # convert to a list of coordinates for each of the 68 landmarks
    shape_transformed = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    point_27_coords = shape_transformed[27]
    point_33_coords = shape_transformed[33]

    distance_27_33 = dist.euclidean(point_27_coords, point_33_coords)

    forehead_coords = (point_27_coords[0],  # X
                       max(0, int(point_27_coords[1] - distance_27_33 / 2)))  # Y

    left_eye_coords = shape_transformed[left_eye_start:left_eye_end]
    right_eye_coords = shape_transformed[right_eye_start:right_eye_end]

    bottom_left_eye_coords = [shape_transformed[i] for i in cfg.FACIAL_LANDMARKS_INDICES['left_eye_bottom']]
    bottom_right_eye_coords = [shape_transformed[i] for i in cfg.FACIAL_LANDMARKS_INDICES['right_eye_bottom']]

    # to capture the eye bag areas we will offset the bottom part of the eye landmarks
    # the offset will be, again, a ratio of the distance_27_33
    bag_offset = int(distance_27_33 * 0.2)

    # append the offset to the coords
    bottom_left_eye_coords_offset = [(landmark[0], landmark[1] + bag_offset)
                                     for landmark in bottom_left_eye_coords]

    bottom_right_eye_coords_offset = [(landmark[0], landmark[1] + bag_offset)
                                      for landmark in bottom_right_eye_coords]
    # captured eye bag regions
    left_bag_hull = cv2.convexHull(np.array(bottom_left_eye_coords + bottom_left_eye_coords_offset))
    right_bag_hull = cv2.convexHull(np.array(bottom_right_eye_coords + bottom_right_eye_coords_offset))

    bag_mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(bag_mask, [left_bag_hull], -1, 255, -1)
    cv2.drawContours(bag_mask, [right_bag_hull], -1, 255, -1)

    forehead_mask = np.zeros(img.shape, np.uint8)
    cv2.circle(forehead_mask, center=forehead_coords, radius=int(distance_27_33 * 0.3), color=255, thickness=-1)

    avg_bag_color = cv2.mean(img, mask=bag_mask)[0]
    avg_skin_color = cv2.mean(img, mask=forehead_mask)[0]

    cv2.drawContours(img, [left_bag_hull], -1, 0, 1)
    cv2.drawContours(img, [right_bag_hull], -1, 0, 1)
    cv2.putText(img, "DIFFERENCE BETWEEN BAGS AND NORMAL SKIN:{:.2f}".format(avg_skin_color - avg_bag_color), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

cv2.imshow('bags', img)
cv2.waitKey(-1)
