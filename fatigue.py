import tensorrt as trt
import torch
import time

# for data process
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_recognizer
from mmcv.tensorrt.tensorrt_utils import (torch_dtype_from_trt,
                                          torch_device_from_trt)

def tensorrt_init_model(ckpt_path):
    """Get predictions by TensorRT engine.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    # load engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(ckpt_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    return engine, context

class FatigueFaceVideoTensorrt():
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path

        # read config file, this is for video dataloader
        self.cfg = mmcv.Config.fromfile(config_path)

        # create engine and context
        self.engine, self.context = tensorrt_init_model(model_path)

        # create ouput
        self.device = torch_device_from_trt(self.engine.get_location(1))
        dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(1))
        output_shape = tuple(self.context.get_binding_shape(1))
        self.output = torch.empty(
            size=output_shape, dtype=dtype, device=self.device, requires_grad=False)

        # Get input tensor shape. For now, only support fixed input tensor
        input_shape = tuple(self.engine.get_binding_shape(0))
        input_dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(0))
        print("InputShape: {}, OutputShape: {}, dtype {}".format(input_shape, output_shape, input_dtype))

        # pipeline
        test_pipeline = self.cfg.data.test.pipeline
        self.test_pipeline = Compose(test_pipeline)

    def __call__(self, face_video_path):
        # get input data
        start_index = self.cfg.data.test.get('start_index', 0)
        data = dict(
            filename=face_video_path,
            label=-1,
            start_index=start_index,
            modality='RGB')
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        # scatter to specified GPU
        data = scatter(data, [self.device])[0]
        data = data['imgs']
        data = data.half()

        # get predictions
        bindings = [
            data.contiguous().data_ptr(),
            self.output.contiguous().data_ptr()
        ]
        #st = time.time()
        self.context.execute_async_v2(bindings,
                                 torch.cuda.current_stream().cuda_stream)
        #print("Time Cost {}".format(time.time() - st))
        results = self.output.cpu().numpy()
        return results