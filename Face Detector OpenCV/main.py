# imports
import cv2

# Code
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

while True:
    #Read current frame
    successfull_frame_read, frame = webcam.read()

    # Convert img colors to grey
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and draw rectangles
    face_coordinates = trained_data.detectMultiScale(grayscaled_img)
    for (x, y, k, l) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + k, y + l), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)

    #Press Q to stop the loop
    if key==81 or key==113:
        break

#Release webcam
webcam.release()
print("End of Code")
