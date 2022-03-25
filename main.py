import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    # cv2.circle(frame, center_bottom,5,(255,255,0),5)

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_only_eye(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
    

    height, width, _ =  frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    cv2.imshow("gray eye", gray_eye)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    
    return threshold_eye

def get_gaze_ratio_ver(eye_points, facial_landmarks):
    threshold_eye = get_only_eye(eye_points, facial_landmarks)
    height, width = threshold_eye.shape
    
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white == 0:
            gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


def get_gaze_ratio_hor(eye_points, facial_landmarks):
    threshold_eye = get_only_eye(eye_points, facial_landmarks)
      
    height, width = threshold_eye.shape
    
    top_side_threshold = threshold_eye[0: int(height / 1.8), 0: width]
    top_side_white = cv2.countNonZero(top_side_threshold)
    top = cv2.resize(top_side_threshold,(500,100))
    cv2.imshow("top eye", top )
    
    bottom_side_threshold = threshold_eye[int(height / 1.8): height, 0 : width]
    h,w =  bottom_side_threshold.shape
    bottom_side_white = cv2.countNonZero(bottom_side_threshold)
    bottom_side_black = h*w -  bottom_side_white
    bot = cv2.resize(bottom_side_threshold,(500,100))
    cv2.imshow("bottom eye", bot)
    
    if bottom_side_white == 0:
        gaze_ratio = 1
    elif bottom_side_black == 0:
        gaze_ratio = 5
  
    
    else:
        gaze_ratio =  bottom_side_white / bottom_side_black
    
    return gaze_ratio


   

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2


        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio_ver([36, 37, 38, 39, 40, 41], landmarks) # for left right
        gaze_ratio_right_eye = get_gaze_ratio_ver([42, 43, 44, 45, 46, 47], landmarks)
        
        gaze_ratio_left_eye1 = get_gaze_ratio_hor([36, 37, 38, 39, 40, 41], landmarks) # for top down
        gaze_ratio_right_eye1 = get_gaze_ratio_hor([42, 43, 44, 45, 46, 47], landmarks)
        
        gaze_ratio1 = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2  # for left right
        gaze_ratio2 = (gaze_ratio_right_eye1 + gaze_ratio_left_eye1) / 2


        cv2.putText(frame, "1 for lr  " + str(gaze_ratio1), (90, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        
        cv2.putText(frame, "2  " + str(gaze_ratio2), (90, 110), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        
        # if gaze_ratio1 <= 0.65:
        #     cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
       
     
            
        # elif 0.65 < gaze_ratio1 < 1.8:
        #     cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        # else:
        
        #     cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
        
        # if gaze_ratio2 >= 1.5:
        #     cv2.putText(frame, "top", (50, 120), font, 2, (0, 0, 255), 3)
            
        
        # if blinking_ratio > 5.7:
        #     cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
        
        # if 5 < blinking_ratio < 5.7:
        #     cv2.putText(frame, "down", (50, 150), font, 7, (255, 0, 0))
        
        #[eye_look_up , eye_look_down, eye_look_left , eye_look_right , eye_blink]
        if blinking_ratio > 5.7:
            cv2.putText(frame, "blink", (50, 100), font, 2, (0, 0, 255), 3)
            gaze= [0,0,0,0,1]
        elif 5.2 < blinking_ratio < 5.7 :
            cv2.putText(frame, "down", (50, 100), font, 2, (0, 0, 255), 3)
            gaze= [0,0,0,0,1]
            
        else:
             
            if gaze_ratio2 >= 1.6:
                cv2.putText(frame, "top", (50, 100), font, 2, (0, 0, 255), 3)
                gaze= [1,0,0,0,0]
            elif gaze_ratio1 <= 0.65:
                cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                gaze= [0,0,0,1,0]
            elif 0.65 < gaze_ratio1 < 2 or gaze_ratio1 < 0.3 :
                cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
                gaze= [0,0,0,0,0]
            else:
        
                cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
                gaze= [0,0,1,0,0]
                
        cv2.putText(frame, str(gaze), (90, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            
        
       
     
            
      
            






    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()