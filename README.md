### Transfering Pytorch Model to C++ Caffe2 Model via onnx
Example how to generate a Pytorch model, use onnx utility to convert them to caffe2 network
Use C++ caffe2 library to predict

-- Run script/pytorch_genmodel.py
-- docker run -it --rm onnx/onnx-docker:cpu /bin/bash
-- convert-onnx-to-caffe2 lenet.onnx --output predict_net.pb --init-net-output init_net.pb
-- Copy predict_net.pb and init_net.pb into example
