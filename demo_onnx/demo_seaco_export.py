"""
Description: 混合模型转onnx格式示例。
Notes: 
- paraformer-zh 是一个多功能的自动语音识别（ASR）模型，可以根据需要选择使用语音活动检测（VAD）、标点符号（punc）和说话人识别（spk）。
- funasr所有模型已移除model revison参数，都不需要指定model revison。
  例如: model = AutoModel(model="paraformer-zh", model_revision="v2.0.4")
- 模型导出路径为: /root/.cache/modelscope/hub/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
"""
from funasr import AutoModel

model = AutoModel(model="paraformer-zh",  vad_model="fsmn-vad", punc_model="ct-punc", disable_update=True,
                  # spk_model="cam++"
                  )

res = model.export(quantize=False)