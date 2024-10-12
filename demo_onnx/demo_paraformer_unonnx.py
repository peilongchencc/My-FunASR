"""
Description: funasr ASR、VAD、PUNC模型组合版运行基础示例。
Notes: 
- 对比版本为onnx代码，即 `demo_seaco_paraformer.py`
"""
from funasr import AutoModel
# paraformer-zh 是一个多功能的自动语音识别（ASR）模型，可以根据需要选择使用语音活动检测（VAD）、标点符号（punc）和说话人识别（spk）。
model = AutoModel(model="paraformer-zh",  
                  vad_model="fsmn-vad", 
                  punc_model="ct-punc", 
                  # spk_model="cam++"
                  disable_update=True,    # 不提示更新funasr
                  )
wav_path = "/data/paraformer/example_wav/asr_example.wav"
res = model.generate(input=wav_path, 
            batch_size_s=300, 
            hotword='魔搭')
print(res)