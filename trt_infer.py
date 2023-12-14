import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import cv2

batch_size = 64
nclass = 5
img_height = 256
img_wdith = 256

def my_softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

def batchSoftmax(x,axis=1):
    # 计算指数
    exp_x = np.exp(x)
    # 对第二维度进行softmax操作
    exp_sum = np.sum(exp_x, axis=axis, keepdims=True)
    softmax_output = exp_x / exp_sum
    return softmax_output

def softmax(x):
    # 将输入展平为二维矩阵（N * H * W，C）
    x_flat = x.reshape((x.shape[0], -1, x.shape[1]))
    # 对每个样本进行softmax计算
    exp_scores = np.exp(x_flat - np.max(x_flat, axis=2, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=2, keepdims=True)
    # 将softmax后的结果重新恢复为原始维度（N，C，H，W）
    probs = probs.reshape(x.shape)
    return probs

# Logger
logger = trt.Logger(trt.Logger.WARNING)

def infer(TRT_MODEL_PATH, input_data):
    with open(TRT_MODEL_PATH, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    # Allocate GPU Memory
    output_shape = (batch_size, nclass, img_height, img_wdith) 
    output_data = np.empty(output_shape, dtype=np.float32)
    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    bindings = [int(d_input), int(d_output)]
    
    # Inference
    with engine.create_execution_context() as context:
        cuda.memcpy_htod(d_input, input_data)
        context.execute(1, bindings)
        cuda.memcpy_dtoh(output_data, d_output)
    
    # Output
    pred = batchSoftmax(output_data)
    result=pred[0] #c, h, w
    seg_result_ori = np.argmax(result, axis=0).astype(np.uint8) #(512, 512)
    seg_result = (seg_result_ori != 0).astype(np.uint8)  # (h, w) # 目前先把所有前景类别当成一类显示
    seg_result_prob = seg_result * result.max(axis=0)
    cv2.imwrite("trt.png", seg_result_prob * 255)
    del engine
    

if __name__ == "__main__":
    # imgPath = "./data/wafer/crop/0000_Row006_Col036_00137_14.bmp"
    imgPath = "./data/wafer/crop/0500_Row023_Col050_00953_21.bmp"


    # TensorRT model - FP32
    fp32 = False
    ptq = False
    qat = True
    if fp32:
        TRT_MODEL_PATH = "./checkpoints/wafer-train.trt"
    if ptq:
        TRT_MODEL_PATH = "./checkpoints/wafer-ptq-calibrated.trt"
    if qat:
        TRT_MODEL_PATH = "./checkpoints/wafer-qat-calibrated.trt"
    # Input: pre-processed feature
    img = cv2.imread(imgPath, 0)
    img = np.expand_dims(img, axis=0)
    img = np.array([img for _ in range(batch_size)])
    # img = np.expand_dims(img, axis=0).astype(np.float32)

    infer(TRT_MODEL_PATH, img)

