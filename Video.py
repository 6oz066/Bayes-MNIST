import cv2

video = cv2.VideoCapture(0)
cnt = 0 # Counter for reading the first cap
while True:
    ret, frame = video.read()
