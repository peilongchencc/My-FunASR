"""
Description: Paraformer(ASR onnx模型)调用示例。
Notes: 
- 混合版onnx模型、ASR onnx模型、VAD onnx模型的调用方式均不同，注意区分。
  详情可查看[FunASR onnx专栏](https://github.com/modelscope/FunASR/tree/main/runtime/python/onnxruntime)。
"""
from funasr_onnx import Paraformer
# model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_dir = "/root/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

model = Paraformer(model_dir, batch_size=1, quantize=True)
wav_path = ['/root/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav']

result = model(wav_path)
print(result)