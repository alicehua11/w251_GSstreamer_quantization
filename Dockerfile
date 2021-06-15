FROM nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3
RUN apt-get update && apt-get install python3-dev git sudo unzip -y
RUN pip3 install -U pip
RUN pip3 install cffi setuptools pillow matplotlib notebook jetson-stats
WORKDIR /app/tf-trt
# To Run : sudo docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-base:r32.4.4

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
