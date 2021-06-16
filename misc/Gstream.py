import numpy as np
import cv2
import time

def send_process():
    video_in = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink')
    video_out = cv2.VideoWriter('appsrc ! queue ! nvvidconv ! omxh264enc insert-vui=1 ! h264parse ! rthph264pay config-interval=1 ! udpsink host=192.168.1.143 port=5000 sync=false', 0, 24, (800,600))
    #video_out = cv2.VideoWriter('appsrc ! queue ! videoconvert ! video/x-raw  ! udpsink host=192.168.1.143 port=5000 sync=false', 0, 24, (800,600))


    if not video_in.isOpened() or not video_out.isOpened():
        print("VideoCapture or VideoWriter not opened")
        exit(0)


    while True:
        ret, frame = video_in.read()
        if not ret: break
        print('writing to video_out')
        video_out.write(frame)
        #cv2.imshow(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_in.release()
    video_out.release()

def receive_process():
    cap_receive = cv2.VideoCapture('udpsrc port=5000 ! caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    while True:
        ret, frame = cap_received.read()
        if not ret: break
        cv2.imshow('received_process', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap_receive.release()

send_process()
time.sleep(5)
receive_process()
