nvcc CUDAprac.cu `pkg-config --cflags --libs opencv` -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -pg -g -std=c++11 -o CUDAprac
