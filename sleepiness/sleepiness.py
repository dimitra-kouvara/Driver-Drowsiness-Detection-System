from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
import sleepiness.config as cfg
from multiprocessing import Process


def sleepiness_app(debug_mode=True, device_id=0):
    # initialize a face detector
    detector = dlib.get_frontal_face_detector()

    # initialize landmark predictor
    predictor = dlib.shape_predictor('landmark_predictor.dat')

    # get the relevant indices from the configuration file
    # Reference https://i.stack.imgur.com/g25oX.png for a landmark mapping
    left_eye_start, left_eye_end = cfg.FACIAL_LANDMARKS_INDICES["left_eye"]
    right_eye_start, right_eye_end = cfg.FACIAL_LANDMARKS_INDICES["right_eye"]
    mouth_start, mouth_end = cfg.FACIAL_LANDMARKS_INDICES["mouth"]
    inner_mouth_start, inner_mouth_end = cfg.FACIAL_LANDMARKS_INDICES["inner_mouth"]

    # start the video stream thread
    cap = cv2.VideoCapture(device_id)

    frame_counter = 0
    alarm_flag = False

    # loop over frames from the video stream
    while True:
        ret, frameC = cap.read()
        frame = cv2.cvtColor(frameC, cv2.COLOR_BGR2GRAY)  # grayscale

        # detect faces
        face_rectangles = detector(frame, 0)

        # loop over the face detections
        for face_rectangle in face_rectangles:
            # determine the facial landmarks for the face region
            shape = predictor(frame, face_rectangle)

            # convert to a list of coordinates for each of the 68 landmarks
            shape_transformed = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

            """
            HEAD TILTING DETECTION
            """

            point_27_coords = shape_transformed[27]
            point_33_coords = shape_transformed[33]

            distance_27_33 = dist.euclidean(point_27_coords, point_33_coords)

            # for head tilting, we need a reference to a constant vertical line
            vertical_line_endpoint = (point_33_coords[0], point_33_coords[1] - int(distance_27_33))

            if debug_mode:
                cv2.line(frameC, point_27_coords, point_33_coords,
                         cfg.FACIAL_LANDMARKS_COLORS['nose1'])  # line on top of nose
                cv2.line(frameC, vertical_line_endpoint, point_33_coords,
                         cfg.FACIAL_LANDMARKS_COLORS['nose2'])  # vertical line starting from nose

            # then we solve the cosine similarity formula by theta
            vector_1 = np.array(point_33_coords) - np.array(point_27_coords)
            vector_2 = np.array(point_33_coords) - np.array(vertical_line_endpoint)

            cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            angle = np.arccos(cosine_angle)  # this is the angle in radians
            head_angle = np.degrees(angle)  # convert radians to degrees - π radians = 180 degrees

            """
            BAG DETECTION
            """

            # we figured, after trial and error, that dividing distance_27_33 by 2 is a good factor to detect the
            # position of the forehead
            # we also use max here to make sure no negative values can be returned (if the forehead falls off the frame)
            forehead_coords = (point_27_coords[0],  # X
                               max(0, int(point_27_coords[1] - distance_27_33/2)))  # Y

            left_eye_coords = shape_transformed[left_eye_start:left_eye_end]
            right_eye_coords = shape_transformed[right_eye_start:right_eye_end]

            bottom_left_eye_coords = [shape_transformed[i] for i in cfg.FACIAL_LANDMARKS_INDICES['left_eye_bottom']]
            bottom_right_eye_coords = [shape_transformed[i] for i in cfg.FACIAL_LANDMARKS_INDICES['right_eye_bottom']]

            # to capture the eye bag areas we will offset the bottom part of the eye landmarks
            # the offset will be, again, a ratio of the distance_27_33
            bag_offset = int(distance_27_33*0.2)

            # append the offset to the coords
            bottom_left_eye_coords_offset = [(landmark[0], landmark[1] + bag_offset)
                                             for landmark in bottom_left_eye_coords]

            bottom_right_eye_coords_offset = [(landmark[0], landmark[1] + bag_offset)
                                              for landmark in bottom_right_eye_coords]
            # captured eye bag regions
            left_bag_hull = cv2.convexHull(np.array(bottom_left_eye_coords + bottom_left_eye_coords_offset))
            right_bag_hull = cv2.convexHull(np.array(bottom_right_eye_coords + bottom_right_eye_coords_offset))

            bag_mask = np.zeros(frame.shape, np.uint8)
            cv2.drawContours(bag_mask, [left_bag_hull], -1, 255, -1)
            cv2.drawContours(bag_mask, [right_bag_hull], -1, 255, -1)

            forehead_mask = np.zeros(frame.shape, np.uint8)
            cv2.circle(forehead_mask, center=forehead_coords, radius=int(distance_27_33*0.3), color=255, thickness=-1)

            avg_bag_color = cv2.mean(frame, mask=bag_mask)[0]
            avg_skin_color = cv2.mean(frame, mask=forehead_mask)[0]

            if debug_mode:
                cv2.drawContours(frameC, [left_bag_hull], -1, cfg.FACIAL_LANDMARKS_COLORS["eyes_bottom"], 1)
                cv2.drawContours(frameC, [right_bag_hull], -1, cfg.FACIAL_LANDMARKS_COLORS["eyes_bottom"], 1)

            """
            EYE CLOSING DETECTION
            """

            # find eye aspect ratios
            left_aspect_ratio = cfg.aspect_ratio(part='eye', point_coords=left_eye_coords)
            right_aspect_ratio = cfg.aspect_ratio(part='eye', point_coords=right_eye_coords)

            # average the eye aspect ratio together for both eyes
            avg_aspect_eye = (left_aspect_ratio + right_aspect_ratio) / 2.0

            if debug_mode:
                left_hull = cv2.convexHull(np.array(left_eye_coords))
                right_hull = cv2.convexHull(np.array(right_eye_coords))
                cv2.drawContours(frameC, [left_hull], -1, cfg.FACIAL_LANDMARKS_COLORS["eyes"], 1)
                cv2.drawContours(frameC, [right_hull], -1, cfg.FACIAL_LANDMARKS_COLORS["eyes"], 1)

            """
            YAWNING DETECTION
            """

            mouth_coords = shape_transformed[mouth_start:mouth_end]
            inner_mouth_coords = shape_transformed[inner_mouth_start:inner_mouth_end]

            # find mouth aspect ratios
            mouth_aspect_ratio = cfg.aspect_ratio(part='mouth', point_coords=mouth_coords)
            inner_mouth_aspect_ratio = cfg.aspect_ratio(part='inner_mouth', point_coords=inner_mouth_coords)

            if debug_mode:
                mouth_hull = cv2.convexHull(np.array(mouth_coords))
                cv2.drawContours(frameC, [mouth_hull], -1, cfg.FACIAL_LANDMARKS_COLORS["mouth"], 1)
                inner_mouth_hull = cv2.convexHull(np.array(inner_mouth_coords))
                cv2.drawContours(frameC, [inner_mouth_hull], -1, cfg.FACIAL_LANDMARKS_COLORS["inner_mouth"], 1)

            """
            THRESHOLDS
            """

            # check to see if the sleepiness indicators get triggered
            # and if so, increment the frame_counter
            trigger_id = None
            if avg_aspect_eye < cfg.EYE_ASPECT_RATIO_THRESH:
                trigger_id = 'E'
            if avg_skin_color - avg_bag_color > cfg.BAG_THRESHOLD:
                trigger_id = 'B'
            if head_angle > cfg.HEAD_ANGLE_THRESHOLD:
                trigger_id = 'H'
            if mouth_aspect_ratio > cfg.MOUTH_ASPECT_RATIO_THRESHOLD or \
                    inner_mouth_aspect_ratio > cfg.INNER_MOUTH_ASPECT_RATIO_THRESHOLD:
                trigger_id = 'M'

            if trigger_id:
                frame_counter += 1

                # if there is an indication for a consecutive number of frames, then sound the alarm
                if frame_counter >= cfg.FRAMES:
                    if not alarm_flag:
                        alarm_flag = True

                        # sound alert on a separate process
                        alarm_process = Process(name="alert", target=cfg.audio_alert, args=(cfg.sounds[trigger_id], ))
                        alarm_process.start()

                    # draw an alert on the bottom of the frame
                    cv2.putText(frameC, "SLEEPINESS INDICATION", (10, frameC.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
            else:  # reset counters
                if alarm_flag:
                    alarm_process.terminate()
                frame_counter = 0
                alarm_flag = False

            # display metrics
            if debug_mode:
                cv2.putText(frameC, "EYE ASPECT RATIO: {:.2f}".format(avg_aspect_eye), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.FACIAL_LANDMARKS_COLORS['eyes'], 1)
                cv2.putText(frameC, "ANGLE: {:.2f}".format(head_angle), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.FACIAL_LANDMARKS_COLORS['nose1'], 1)
                cv2.putText(frameC, "DIFFERENCE BETWEEN BAGS AND NORMAL SKIN:{:.2f}".format(avg_skin_color - avg_bag_color), (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.FACIAL_LANDMARKS_COLORS['eyes_bottom'], 1)
                cv2.putText(frameC, "MOUTH ASPECT RATIO:{:.2f}".format(mouth_aspect_ratio), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.FACIAL_LANDMARKS_COLORS['mouth'], 1)
                cv2.putText(frameC, "INNER MOUTH ASPECT RATIO: {:.2f}".format(inner_mouth_aspect_ratio), (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.FACIAL_LANDMARKS_COLORS['inner_mouth'], 1)
        cv2.imshow('Webcam', frameC)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # close webcam stream
    cap.release()
    cv2.destroyAllWindows()

