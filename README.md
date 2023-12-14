# QAT demo
QAT(Quantization Aware Training)示例, 使用[pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)库

所有资源链接：https://pan.baidu.com/s/1cuBdabfEqqsA62Qt0g8DiQ?pwd=a00d 

## 使用过程
```shell
QAT步骤

1. Insert Q&DQ nodes to get fake-quant pytorch model
2. PTQ calibration
3. QAT Training



代码运行步骤

1. 准备环境镜像, 启动容器
要补充安装几个库
$ pip install pytorch-quantization==2.1.2 --extra-index-url https://pypi.ngc.nvidia.com
$ pip install ppq
$ pip install pycuda
$ apt-get install ninja-build

2. 上传数据到对应位置 

3. 执行训练和验证 
$ python train.py
$ cp checkpoints/wafer-train/某一个pth  checkpoints/wafer-train.pth
$ python train-predict.py
$ python train-convert_onnx.py

4. 执行PTQ内容
$ python ptq.py
$ python ptq-predict.py

5. 执行QAT
$ python qat.py
$ cp checkpoints/wafer-qat/某一个pth  checkpoints/wafer-qat-calibrated.pth
$ python qat-predict.py
$ python ptq_qat-convert_onnx.py

6. 转trt
$ python convert_trt.py

7. trt推理
$ python trt_infer.py
```

```shell
ppq库
$ python ppq-convert_onnx.py
$ python ppq-trt_infer.py
```