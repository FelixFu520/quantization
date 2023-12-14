import tensorrt as trt
import os

from utils import Calibrator


LOGGER = trt.Logger(trt.Logger.VERBOSE)

def buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                img_height, img_wdith, calibration_images, calibration_cache):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16*(1<<20))

    if FP16_mode == True:
        config.set_flag(trt.BuilderFlag.FP16)
    elif INT8_mode == True:
        print("start INT8 Calibrator")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(quantification, batch_size, img_height, img_wdith, calibration_images, calibration_cache)
    engine = builder.build_serialized_network(network, config)
    print("Completed creating Engine")
    if engine is None:
        print("EXPORT ENGINE FAILED!")
        exit(0)

    with open(engine_file, "wb") as f:
        f.write(engine)


def main():
    quantification = 1  # quantization times
    batch_size = 64
    img_height = 256
    img_wdith = 256

    # 转FP32 onnx
    convert_fp32 = False
    if convert_fp32:
        onnx_file = "./checkpoints/wafer-train.onnx"
        engine_file = "./checkpoints/wafer-train.trt"
        FP16_mode = True
        INT8_mode = False
        if not os.path.exists(onnx_file):
            print("LOAD ONNX FILE FAILED: ", onnx_file)

        print('Load ONNX file from:%s \nStart export, Please wait a moment...' % (onnx_file))
        buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                    img_height, img_wdith, "", "")
        print('Export ENGINE success, Save as: ', engine_file)

    # 转ptq onnx
    convert_ptq = False
    if convert_ptq:
        calibration_images = "data/wafer/calib"
        onnx_file = "./checkpoints/wafer-ptq-calibrated.onnx"
        engine_file = "./checkpoints/wafer-ptq-calibrated.trt"
        calibration_cache = "./checkpoints/wafer-ptq-calibrated.cache"

        FP16_mode = False
        INT8_mode = True

        if not os.path.exists(onnx_file):
            print("LOAD ONNX FILE FAILED: ", onnx_file)

        print('Load ONNX file from:%s \nStart export, Please wait a moment...' % (onnx_file))
        buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                    img_height, img_wdith, calibration_images, calibration_cache)
        print('Export ENGINE success, Save as: ', engine_file)

    # 转qat onnx
    convert_qat = True
    if convert_qat:
        calibration_images = "data/wafer/calib"
        onnx_file = "./checkpoints/wafer-qat-calibrated.onnx"
        engine_file = "./checkpoints/wafer-qat-calibrated.trt"
        calibration_cache = "./wafer-qat-calibrated.cache"

        FP16_mode = False
        INT8_mode = True

        if not os.path.exists(onnx_file):
            print("LOAD ONNX FILE FAILED: ", onnx_file)

        print('Load ONNX file from:%s \nStart export, Please wait a moment...'%(onnx_file))
        buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                    img_height, img_wdith, calibration_images, calibration_cache)
        print('Export ENGINE success, Save as: ', engine_file)


if __name__ == '__main__':
    main()

