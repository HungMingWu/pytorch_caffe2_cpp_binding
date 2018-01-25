all:
	g++ main.cpp mnist.cpp -g -O0 -o main -std=c++11 -lprotobuf -lcaffe2

clean:
	rm main
