# W251 - Summmer 2021 - Homework 6
### Section 2: Alice Hua

This repository contains my homework 6 **Optimizing Models for the Edge and GStreamer** for course W251 - Deep Learning in the Cloud and at the Edge at the UC Berkeley School of Information.

### Part I: GStreamer 
**Note**: you can use `jtop` to see Jetson hardware accelerators. NVDEC and NVENC are the GPU hardware accelerator engines for video decoding that support transcoding application.

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
Python source code: look at client.py

### Part 2: Model optimization and quantization
Similar to lab, we converted Keras saved model to TensorRT to look at

**Note**: If run on Docker container, make sure to raise the shared memory limit so that you can use higher batch size. You can do so with the command for JetPack version 4.4.1 or can be found at this [link](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md)
```
docker run --gpus all -it --rm --shm-size=2048m dustynv/jetson-inference:r32.4.4
```
 

1) The base model used, trained using Jetson device and Jetson Inference scripts:
- mobilenet-v1-ssd-mp-0_675.pth

2) A description of your dataset 
Using the [Google Open Images Dataset](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation), I chose seven classes of animals. The below command will download the dataset onto the running container:
```
python3 open_images_downloader.py --class-names "Rhinoceros,Sea turtle,Sparrow,Whale,Zebra,Blue jay,Armadillo" --data=data/animal
``` 
A total of 4300 images of 7 classes were downloaded. Below is the stats for the train/validation/test dataset
```
-------------------------------------
 'train' set statistics
-------------------------------------
  Image count:  3626
  Bounding box count:  5179
  Bounding box distribution: 
    Sparrow:  1387/5179 = 0.27
    Zebra:  993/5179 = 0.19
    Sea turtle:  958/5179 = 0.18
    Whale:  912/5179 = 0.18
    Rhinoceros:  659/5179 = 0.13
    Blue jay:  232/5179 = 0.04
    Armadillo:  38/5179 = 0.01
 

-------------------------------------
 'validation' set statistics
-------------------------------------
  Image count:  171
  Bounding box count:  220
  Bounding box distribution: 
    Sparrow:  63/220 = 0.29
    Zebra:  41/220 = 0.19
    Whale:  41/220 = 0.19
    Sea turtle:  33/220 = 0.15
    Rhinoceros:  21/220 = 0.10
    Blue jay:  21/220 = 0.10
 

-------------------------------------
 'test' set statistics
-------------------------------------
  Image count:  505
  Bounding box count:  675
  Bounding box distribution: 
    Sparrow:  159/675 = 0.24
    Sea turtle:  152/675 = 0.23
    Whale:  124/675 = 0.18
    Zebra:  111/675 = 0.16
    Rhinoceros:  69/675 = 0.10
    Blue jay:  60/675 = 0.09
 

-------------------------------------
 Overall statistics
-------------------------------------
  Image count:  4302
  Bounding box count:  6074

```

 
3) How long 
|           | Images/sec | Time per epoch* |
| Xavier NX |            |                 |

Below are some of the images output of this model. Some were not classified at all while some did.
![](misc/results.png)

```
[TRT]    device GPU, models/animal/ssd-mobilenet.onnx initialized.
[TRT]    detectNet -- number object classes:  8
[TRT]    detectNet -- maximum bounding boxes:  3000
[TRT]    detectNet -- loaded 8 class info entries
[TRT]    detectNet -- number of object classes:  8
[image] loaded '/usr/local/bin/images/img0.JPG'  (2048x1869, 3 channels)
5 objects detected
detected obj 0  class #3 (Rhinoceros)  confidence=0.970356
bounding box 0  (0.000000, 965.295959)  (1609.607300, 1776.130249)  w=1609.607300  h=810.834290
detected obj 1  class #3 (Rhinoceros)  confidence=0.706137
bounding box 1  (0.000000, 1012.453369)  (1720.677124, 1776.445068)  w=1720.677124  h=763.991699
detected obj 2  class #3 (Rhinoceros)  confidence=0.532596
bounding box 2  (646.110840, 1108.035645)  (1851.313843, 1700.723755)  w=1205.203003  h=592.688110
detected obj 3  class #3 (Rhinoceros)  confidence=0.600264
bounding box 3  (885.490173, 1126.113281)  (1794.356445, 1637.055908)  w=908.866272  h=510.942627
detected obj 4  class #3 (Rhinoceros)  confidence=0.629085
bounding box 4  (942.112732, 1201.857178)  (1824.761719, 1671.200317)  w=882.648987  h=469.343140
[image] saved 'test_animal/0.jpg'  (2048x1869, 3 channels)

[TRT]    ------------------------------------------------
[TRT]    Timing Report models/animal/ssd-mobilenet.onnx
[TRT]    ------------------------------------------------
[TRT]    Pre-Process   CPU   0.39181ms  CUDA   0.30445ms
[TRT]    Network       CPU   9.21164ms  CUDA   9.24570ms
[TRT]    Post-Process  CPU   1.25921ms  CUDA   1.14074ms
[TRT]    Visualize     CPU  67.38385ms  CUDA  68.84650ms
[TRT]    Total         CPU  78.24651ms  CUDA  79.53738ms
[TRT]    ------------------------------------------------

[TRT]    note -- when processing a single image, run 'sudo jetson_clocks' before
                to disable DVFS for more accurate profiling/timing measurements

[image] loaded '/usr/local/bin/images/img1.JPG'  (1227x1879, 3 channels)
[image] saved 'test_animal/1.jpg'  (1227x1879, 3 channels)

[TRT]    ------------------------------------------------
[TRT]    Timing Report models/animal/ssd-mobilenet.onnx
[TRT]    ------------------------------------------------
[TRT]    Pre-Process   CPU   0.08944ms  CUDA   0.11878ms
[TRT]    Network       CPU   6.59537ms  CUDA   6.43939ms
[TRT]    Post-Process  CPU   0.17853ms  CUDA   0.17920ms
[TRT]    Visualize     CPU  67.38385ms  CUDA  68.84650ms
[TRT]    Total         CPU  74.24718ms  CUDA  75.58387ms
[TRT]    ------------------------------------------------

[image] loaded '/usr/local/bin/images/img2.JPG'  (1234x1851, 3 channels)
2 objects detected
detected obj 0  class #5 (Sparrow)  confidence=0.622882
bounding box 0  (204.571365, 699.187073)  (694.416931, 1354.285278)  w=489.845581  h=655.098206
detected obj 1  class #5 (Sparrow)  confidence=0.510817
bounding box 1  (321.960541, 768.955750)  (726.457397, 1163.386963)  w=404.496857  h=394.431213
[image] saved 'test_animal/2.jpg'  (1234x1851, 3 channels)

[TRT]    ------------------------------------------------
[TRT]    Timing Report models/animal/ssd-mobilenet.onnx
[TRT]    ------------------------------------------------
[TRT]    Pre-Process   CPU   0.12509ms  CUDA   0.11120ms
[TRT]    Network       CPU   3.27080ms  CUDA   3.04947ms
[TRT]    Post-Process  CPU   0.16160ms  CUDA   0.16179ms
[TRT]    Visualize     CPU   0.17350ms  CUDA   0.92566ms
[TRT]    Total         CPU   3.73100ms  CUDA   4.24813ms
[TRT]    ------------------------------------------------

[image] loaded '/usr/local/bin/images/img3.JPG'  (2094x1402, 3 channels)
[image] saved 'test_animal/3.jpg'  (2094x1402, 3 channels)

[TRT]    ------------------------------------------------
[TRT]    Timing Report models/animal/ssd-mobilenet.onnx
[TRT]    ------------------------------------------------
[TRT]    Pre-Process   CPU   0.07373ms  CUDA   0.13517ms
[TRT]    Network       CPU   3.76050ms  CUDA   3.45616ms
[TRT]    Post-Process  CPU   0.15216ms  CUDA   0.15258ms
[TRT]    Visualize     CPU   0.17350ms  CUDA   0.92566ms
[TRT]    Total         CPU   4.15990ms  CUDA   4.66957ms
[TRT]    ------------------------------------------------

[image] loaded '/usr/local/bin/images/img4.JPG'  (2078x1184, 3 channels)
5 objects detected
detected obj 0  class #7 (Zebra)  confidence=0.717969
bounding box 0  (566.541992, 147.395752)  (985.949951, 592.689636)  w=419.407959  h=445.293884
detected obj 1  class #7 (Zebra)  confidence=0.871105
bounding box 1  (1504.275146, 210.863770)  (1937.412354, 615.475159)  w=433.137207  h=404.611389
detected obj 2  class #7 (Zebra)  confidence=0.565797
bounding box 2  (617.645630, 187.777618)  (982.454956, 558.003601)  w=364.809326  h=370.225983
detected obj 3  class #7 (Zebra)  confidence=0.750679
bounding box 3  (1037.541992, 120.939011)  (1334.706055, 572.811035)  w=297.164062  h=451.872009
detected obj 4  class #7 (Zebra)  confidence=0.691800
bounding box 4  (642.098694, 779.244263)  (905.240173, 1040.693237)  w=263.141479  h=261.448975
[image] saved 'test_animal/4.jpg'  (2078x1184, 3 channels)

[TRT]    ------------------------------------------------
[TRT]    Timing Report models/animal/ssd-mobilenet.onnx
[TRT]    ------------------------------------------------
[TRT]    Pre-Process   CPU   0.07383ms  CUDA   0.14234ms
[TRT]    Network       CPU   6.25236ms  CUDA   5.96934ms
[TRT]    Post-Process  CPU   0.17731ms  CUDA   0.17715ms
[TRT]    Visualize     CPU   0.19098ms  CUDA   1.14774ms
[TRT]    Total         CPU   6.69447ms  CUDA   7.43658ms
[TRT]    ------------------------------------------------

[image] loaded '/usr/local/bin/images/img5.JPG'  (2100x1400, 3 channels)
2 objects detected
detected obj 0  class #5 (Sparrow)  confidence=0.711948
bounding box 0  (977.580933, 342.702240)  (1518.219849, 1285.143066)  w=540.638916  h=942.440796
detected obj 1  class #5 (Sparrow)  confidence=0.738255
bounding box 1  (689.878235, 434.400055)  (1078.404541, 896.520020)  w=388.526306  h=462.119965
[image] saved 'test_animal/5.jpg'  (2100x1400, 3 channels)

[TRT]    ------------------------------------------------
[TRT]    Timing Report models/animal/ssd-mobilenet.onnx
[TRT]    ------------------------------------------------
[TRT]    Pre-Process   CPU   0.07664ms  CUDA   0.13331ms
[TRT]    Network       CPU   3.65301ms  CUDA   3.36922ms
[TRT]    Post-Process  CPU   0.30951ms  CUDA   0.30822ms
[TRT]    Visualize     CPU   0.18135ms  CUDA   1.09648ms
[TRT]    Total         CPU   4.22051ms  CUDA   4.90723ms
[TRT]    ------------------------------------------------

[image] loaded '/usr/local/bin/images/img6.JPG'  (2048x907, 3 channels)
[image] saved 'test_animal/6.jpg'  (2048x907, 3 channels)

[TRT]    ------------------------------------------------
[TRT]    Timing Report models/animal/ssd-mobilenet.onnx
[TRT]    ------------------------------------------------
[TRT]    Pre-Process   CPU   0.08170ms  CUDA   0.13661ms
[TRT]    Network       CPU   3.41333ms  CUDA   3.13037ms
[TRT]    Post-Process  CPU   0.27255ms  CUDA   0.27187ms
[TRT]    Visualize     CPU   0.18135ms  CUDA   1.09648ms
[TRT]    Total         CPU   3.94892ms  CUDA   4.63533ms
[TRT]    ------------------------------------------------

[image] imageLoader -- End of Stream (EOS) has been reached, stream has been closed
detectnet:  shutting down...
detectnet:  shutdown complete.

```
### References:
- For GStreamer:
	- https://docs.nvidia.com/metropolis/deepstream/DeepStream_5.0.1_Release_Notes.pdf
	- https://forums.developer.nvidia.com/t/how-to-stream-frames-using-gstreamer-with-opencv-in-python/121036/3
	- https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html
	- https://medium.com/@fanzongshaoxing/use-nvidia-deepstream-to-accelerate-h-264-video-stream-decoding-8f0fec764778
- For Jetson Inference & Optimization
	- https://github.com/dusty-nv/jetson-inference
	- https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md
	- https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt
	- https://storage.googleapis.com/openimages/web/download.html
	- https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
	 
