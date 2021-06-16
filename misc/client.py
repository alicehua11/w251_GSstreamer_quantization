import numpy as np
import cv2

# use gstreamer for video directly; set the fps
#camSet='v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink'
#camSet='v4l2src device=/dev/video0  ! nvvidconv ! omxh265enc insert-vui=1 ! h265parse ! rtph265pay config-interval=1 ! udpsink host=192.168.1.143 port=5000'
#cap= cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)

#out = cv2.VideoWriter('rtspsrc ! nvvidconv ! omxh265enc insert-vui=1 ! h265parse ! rtph265pay config-interval=1 ! udpsink host=192.168.1.143 port=5000', cv2.CAP_GSTREAMER, 0, 25.0, (1920,1080))



def send_process():
    video_in = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink')
    video_out = cv2.VideoWriter('appsrc ! queue ! videoconvert ! video/x-raw ! nvvidconv ! omxh264enc insert-vui=1 ! h264parse !  rtph264pay config-interval=1 ! udpsink host=192.168.1.143 port=5000', 0, 24, (800,600))
   # video_out = cv2.VideoWriter('appsrc ! queue ! videoconvert ! videoscale ! video/x-raw,format=I420,width=1080,height=240,framerate=30/1 !  videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=192.168.1.143 port=5000 sync=false', 0, 24, (800,600))

    if not video_in.isOpened() or not video_out.isOpened():
        print("VideoCapture or VideoWriter not opened")
        exit(0)


    while True:
        ret, frame = video_in.read()
        if not ret: break
        #print('writing to video_out')
        video_out.write(frame)
        cv2.imshow('client', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_in.release()
    video_out.release()


send_process()
cv2.destroyAllWindows()
