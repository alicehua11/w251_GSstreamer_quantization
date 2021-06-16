# W251 - Summmer 2021 - Homework 3
### Section 2: Alice Hua

This repository contains my homework 6 **Optimizing Models for the Edge and GStreamer** for course W251 - Deep Learning in the Cloud and at the Edge at the UC Berkeley School of Information.

### Part I: GStreamer 
* Note: you can use `jtop` to see Jetson hardware accelerators. NVDEC and NVENC are the GPU hardware accelerator engines for video decoding that support transcoding application.

1. Convert this pipeline to use the Nvidia nveglglesink.
```
# Nvidia sink nv3dsink
gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink

# Nvidia sink nveglglessink
gst-launch-1.0 v4l2src device=/dev/video0 ! nvvidconv ! nvegltransform ! nveglglessink -e

# or
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1 ! nvvidconv ! nvegltransform ! nveglglessink -e 
``` 

2. Explain the difference between a property and a capability? How are they each expressed in a pipeline?
- A property is used to configure or to modify the behavior of an element, separated by spaces. A capacitity is describes the ypes of media that may stream over a pad, separated by commas (a pad is an element's interace to the outside world).

3. Explain the following pipeline, that is explain each piece of the pipeline, desribing if it is an element (if so, what type), property, or capability. What does this pipeline do?
```
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1 ! videoconvert ! agingtv scratch-lines=10 ! videoconvert ! xvimagesink sync=false
```

- This is opening the tool gst-launch-1.0 to run a pipeline from source element video4linux from a device location at /dev/video0 and ask for x-raw format with a frame rate of 30 fps then convert it to a format that can receive a special agingtv effect with property scratch-lines of 10 then convert it to a final format to send to sink element that gives us the video output without syncing on the clock.

4) GStreamer pipelines may also be used from Python and OpenCV. Write a Python application that listens for images streamed from a GStreamer pipeline.
Broadcaster/client GStreamer pipeline
```
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1,width=640,height=480  ! nvvidconv ! omxh265enc insert-vui=1 ! h265parse ! rtph265pay config-interval=1 ! udpsink host=192.168.1.143 port=5000 sync=false -e 
```
Python source code: look at server.py

### Part 2: Model optimization and quantization
Similar to lab, we converted Keras saved model to TensorRT to look at 



### References:
- For GStreamer:
	- https://docs.nvidia.com/metropolis/deepstream/DeepStream_5.0.1_Release_Notes.pdf
	- https://forums.developer.nvidia.com/t/how-to-stream-frames-using-gstreamer-with-opencv-in-python/121036/3
	- https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html
	- https://medium.com/@fanzongshaoxing/use-nvidia-deepstream-to-accelerate-h-264-video-stream-decoding-8f0fec764778
- For Jetson Inference & Optimization
	- https://github.com/dusty-nv/jetson-inference
	- https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md
	- https://storage.googleapis.com/openimages/web/download.html
	- https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
	 
