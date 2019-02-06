# This is a CNN implementation using cuDNN in CUDA C++

The aim of this implementation is to enable interactive learning for an autonomous driving application, which will be trained by an individual in real-time with images received while driving. This network can map human behaviour of driving using the input image feed and corresponding turning radius provided by the driver. 

A very big aspect of training this network is dataset that can map the behaviour aptly.

Devoloping in CUDA C++ rather than Python can provide the platform for online training, although is not necessary. Once finished, this network can produce driving angle using just the image feed, with no external controllers or object localization 
algorithms (like YOLO or SSD) needed. 
