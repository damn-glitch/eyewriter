import cv2
import dlib
import numpy as np
from eye_key_funcs import *
from projected_keyboard import *


# Set the camera ID
camera_ID = 0

# Initialize camera
camera = init_camera(camera_ID)

# Create a black page for calibration (adjust the size as needed)
calibration_page = make_black_page((800, 600))

# Initialize face and landmark detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Perform calibration (adjust as needed)
calibration_cut = [(100, 100), (700, 100), (700, 500), (100, 500)]

# Get keyboard information
width_keyboard = 800
height_keyboard = 400
offset_keyboard = (50, 150)
keys = get_keyboard(width_keyboard, height_keyboard, offset_keyboard)

# Initialize typing variables
selected_key = None
typed_word = ""

# Main loop
while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Adjust the frame if needed
    frame = adjust_frame(frame)

    # Detect faces in the frame
    faces = detector(frame)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(frame, face)

        # Display a box around the face
        display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (10, 10))

        # Get eye coordinates
        eye_coordinates = get_eye_coordinates(landmarks, [36, 37, 38, 39, 40, 41])

        # Display lines on the eyes
        display_eye_lines(frame, eye_coordinates, 'blue')

        # Display face points
        display_face_points(frame, landmarks, [36, 42], 'red')

        # Identify the currently selected key based on eye gaze
        coordinate_X = half_point_x(landmarks.part(36), landmarks.part(39))
        coordinate_Y = half_point(landmarks.part(36), landmarks.part(39))[1]
        selected_key = identify_key(keys, coordinate_X, coordinate_Y)

        # Highlight the selected key
        if selected_key and len(selected_key) >= 4:
            cv2.rectangle(frame, tuple(map(int, selected_key[2])), tuple(map(int, selected_key[3])), (0, 255, 0), thickness=2)

        # Check for blinking to confirm letter selection
        if is_blinking(eye_coordinates):
            print(f"Blink detected. Selected key: {selected_key}")
            if selected_key:
                typed_word += selected_key[0]
                print(f"Typed word: {typed_word}")
                selected_key = None

    # Display the keyboard on the frame
    display_keyboard(frame, keys)

    # Display the frame
    cv2.imshow("Eye Tracking", frame)

    # Check for the 'Esc' key to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close OpenCV windows
shut_off(camera)
cv2.destroyAllWindows()