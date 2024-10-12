"""
Description: 单独模型转onnx格式示例。
Notes: 
"""
from funasr import AutoModel

model = AutoModel(model="paraformer", disable_update=True)
res = model.export(quantize=False)
# output dir: ～/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

model = AutoModel(model="fsmn-vad", disable_update=True)
res = model.export(quantize=False)
# output dir: ～/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch

model = AutoModel(model="ct-punc-c", disable_update=True)
res = model.export(quantize=False)
# output dir: ～/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch



