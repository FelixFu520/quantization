# 量化学习
复现中所需要的文件都放到了百度云上

链接: https://pan.baidu.com/s/1y25msMxuFqrh7ijuUsnZTw?pwd=8ssq

## 开发环境
从百度上下载TensorRT资源包, 放到项目根目录下, 然后运行以下命令
```
sudo docker build -t quantization:latest . -f DockerFile
sudo docker run -p 10022:22 --name quantization -itd -v /data:/root/data --gpus all --privileged --shm-size=64g quantization:latest

pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install setuptools==59.5.0 torchsummary -i https://pypi.tuna.tsinghua.edu.cn/simple

```
## 准备数据集
从百度云上下载数据集, 然后解压至项目根目录

## 使用
```shell
1. train.py
2. infer.py
3. ptq.py
4. qat.py
5. onnx_export.py
6. trt_infer.py
7. trt_infer_acc.py
```


## 参考
- https://github.com/yester31/Quantization_EX
