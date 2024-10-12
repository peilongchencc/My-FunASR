"""
Description: funasr_onnx ASR、VAD、PUNC模型组合版运行bytes转np示例。
Notes: 
运行代码后，会自动下载模型并将模型转化为onnx格式。模型默认保存的路径:
~/.cache/modelscope/hub/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

onnx模型不接受bytes类型数据，funasr_onnx 的 paraformer_bin 中只支持传入[str,np.ndarray,list]。
"""
from pydub import AudioSegment
from funasr_onnx import SeacoParaformer
import numpy as np

# 定义函数：使用 pydub 将 wav 文件转换为 numpy 数组格式
def wav_to_numpy(wav_path: str) -> np.ndarray:
    # 使用 pydub 读取 wav 文件
    audio = AudioSegment.from_wav(wav_path)
    # 将音频数据转换为 numpy 数组
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # 归一化到 [-1.0, 1.0] 范围
    return audio_data

# 加载 SeacoParaformer 模型
model_dir = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = SeacoParaformer(model_dir, batch_size=1, device_id=0)

# 音频文件路径
wav_path = "/data/paraformer/example_wav/asr_example.wav"

# 将音频文件转换为 numpy 数组格式
wav_numpy = wav_to_numpy(wav_path)

# 设置热词（如果有）
hotwords = "你的热词 魔搭"

# 使用模型进行预测
result = model(wav_numpy, hotwords)

# 输出结果
print(result)
