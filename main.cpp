#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <caffe2/core/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

static caffe2::NetDef initNet, predictNet;
static caffe2::Predictor *predictor;

void ReadMNIST(int NumberOfImages, int DataOfAnImage, std::vector<std::vector<float>> &arr);

int testNo = 5000;

void loadToNetDef(caffe2::NetDef* net, const char *filename) {
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		printf("Couldn't open file %s\n", filename);
		exit(-1);
	}
	fseek(fp, 0, SEEK_END);
	size_t size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	std::vector<char> data(size);
	fread(data.data(), data.size(), 1, fp);
	if (!net->ParseFromArray(data.data(), data.size())) {
		printf("Couldn't parse net from data.\n");
		exit(-1);
	}
	fclose(fp);
}

int main()
{
	try {
		std::vector<std::vector<float>> arr;
		ReadMNIST(10000, 784, arr);
		caffe2::TensorCPU input;
		input.Resize(std::vector<int>({1, 1, 28, 28}));
		memcpy(input.mutable_data<float>(), arr[testNo].data(), 28 * 28 * sizeof(float));
#if TEST
		for (size_t i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				if (arr[testNo][i * 28 + j] > 0) printf("1 ");
				else printf("0 ");
			}
			printf("\n");
		}
#endif
		caffe2::Predictor::TensorVector input_vec{&input};
		loadToNetDef(&initNet, "init_net.pb");
		loadToNetDef(&predictNet, "predict_net.pb");
		predictor = new caffe2::Predictor(initNet, predictNet);

		caffe2::Predictor::TensorVector output_vec;
		predictor->run(input_vec, &output_vec);
		
		if (output_vec.capacity() > 0) {
			for (auto output : output_vec) {
				std::cout << "output->size = " << output->size() << "\n"; 
				for (auto i = 0; i < output->size(); ++i) {
					std::cout << i << ": " << output->template data<float>()[i] << "\n";
				}
			}
		} else {
			std::cout << "Empty\n";
		}
	} catch(std::exception &e) {
		std::cout << "Throw exception: " << e.what() << "\n";
	}


}
