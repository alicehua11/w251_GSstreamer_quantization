import numpy as np
import cv2



def receive_process():
   # cap_receive = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp, media=video ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
   # cap_receive = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
   # cap_receive = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw ! videoconvert ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER) 
   # cap_receive = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    cap_receive = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp, media=video, payload=96, clock-rate=90000, encoding-name=H265 ! rtph265depay ! h265parse ! omxh265dec ! videoconvert ! appsink')
    while True:
        ret, frame = cap_receive.read()
        #print('Video received')
        if not ret: break
        cv2.imshow('server', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap_receive.release()


receive_process()
cv2.destroyAllWindows()

