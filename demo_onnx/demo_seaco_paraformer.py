"""
Description: funasr_onnx ASR、VAD、PUNC模型组合版运行基础示例。
Notes: 
- 运行代码后，会自动下载模型并将模型转化为onnx格式。模型默认保存的路径:
- ~/.cache/modelscope/hub/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
- 对比版本为非onnx代码，即 `demo_paraformer_unonnx.py`
"""
import time
from funasr_onnx import SeacoParaformer

# 加载模型
model_dir = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = SeacoParaformer(model_dir, batch_size=1, device_id=0, disable_update=True)

# 音频文件路径
wav_path = "/data/paraformer/example_wav/asr_example.wav"

# 热词
hotwords = "你的热词 魔搭"

# 开始计时
start_time = time.time()

# 进行模型预测
result = model(wav_path, hotwords)

# 结束计时
end_time = time.time()

# 计算并输出预测时间
print(f"模型预测时间：{end_time - start_time:.4f} 秒")
print(result)
# 去除结果中的空格
print("".join(result[0]['raw_tokens']))
