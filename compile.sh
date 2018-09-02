nvcc myCUDNN.cu `pkg-config --cflags --libs opencv` -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -pg -std=c++11 -o myCUDNN

sleep 1

./myCUDNN image.png


