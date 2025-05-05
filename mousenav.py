import cv2
import dlib
import pyautogui
import time
import numpy as np

# Load face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor.dat")

# Initialize camera
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Variables for click detection
eyes_closed = False
blink_start_time = 0
click_threshold = 0.12  # Blink duration for left click
right_click_threshold = 0.6  # Longer blink for right click
file_close_threshold = 1.5  # Blink + head move for file close
smoothing_factor = 5
prev_x, prev_y = None, None

# Head movement detection
head_move_check = False
head_movement_start = 0

def get_eye_ratio(eye_points, facial_landmarks):
    left_point = facial_landmarks.part(eye_points[0])
    right_point = facial_landmarks.part(eye_points[3])
    top_mid = (facial_landmarks.part(eye_points[1]).y + facial_landmarks.part(eye_points[2]).y) / 2
    bottom_mid = (facial_landmarks.part(eye_points[5]).y + facial_landmarks.part(eye_points[4]).y) / 2
    eye_ratio = abs(top_mid - bottom_mid) / abs(left_point.x - right_point.x)
    return eye_ratio

def get_head_orientation(nose, center_x):
    if nose.x < center_x - 40:
        return 'left'
    elif nose.x > center_x + 40:
        return 'right'
    else:
        return 'center'

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    frame_h, frame_w, _ = frame.shape

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_eye_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_eye_ratio([42, 43, 44, 45, 46, 47], landmarks)
        
        # Eye tracking for cursor movement
        nose = landmarks.part(30)
        screen_x = screen_w * (nose.x / frame_w)
        screen_y = screen_h * (nose.y / frame_h)
        
        # Smooth movement
        if prev_x is not None and prev_y is not None:
            screen_x = (screen_x + prev_x * (smoothing_factor - 1)) / smoothing_factor
            screen_y = (screen_y + prev_y * (smoothing_factor - 1)) / smoothing_factor
        pyautogui.moveTo(screen_x, screen_y)
        prev_x, prev_y = screen_x, screen_y

        # Blink detection for clicks and file close
        if left_eye_ratio < 0.2 and right_eye_ratio < 0.2:
            if not eyes_closed:
                eyes_closed = True
                blink_start_time = time.time()
                head_move_check = False
        else:
            if eyes_closed:
                blink_duration = time.time() - blink_start_time
                orientation = get_head_orientation(nose, frame_w // 2)

                if click_threshold < blink_duration < right_click_threshold:
                    pyautogui.click()
                elif right_click_threshold <= blink_duration < file_close_threshold:
                    pyautogui.rightClick()
                elif blink_duration >= file_close_threshold and orientation == 'left':
                    pyautogui.hotkey('alt', 'f4')  # Close current active window
                eyes_closed = False

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or not cam.isOpened():
        break

cam.release()
cv2.destroyAllWindows()
