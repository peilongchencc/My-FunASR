"""
Description: paraformer流式输出示例。
Notes: 
"""
from funasr import AutoModel

# 设置音频分块大小，单位为毫秒
# [0, 10, 5]表示分块的起始、结束和步长，600ms
# [0, 8, 4]表示480ms的配置
chunk_size = [0, 10, 5]

# 编码器自注意力机制查看的分块数量
# 在生成当前分块的输出时，编码器将查看过去的4个分块
encoder_chunk_look_back = 4 

# 解码器交叉注意力机制查看的编码器分块数量
# 在生成当前分块的输出时，解码器将查看过去的1个编码器分块
decoder_chunk_look_back = 1 

# 加载指定模型(不显示funasr更新提示)
# model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", disable_update=True)
model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

import soundfile
import os

# 读取示例音频文件的路径
wav_file = os.path.join(model.model_path, "example/asr_example.wav")

# 使用soundfile库读取音频文件，并获取音频数据和采样率
speech, sample_rate = soundfile.read(wav_file)

# 计算音频分块的步长，乘以960以转换为帧数（假设每帧为1ms）
chunk_stride = chunk_size[1] * 960  # 600ms

# 初始化缓存字典，存储模型推理的中间状态
cache = {}

# 计算音频数据中可以切割出的总分块数量
total_chunk_num = int(len((speech)-1)/chunk_stride+1)

# 遍历每个分块
for i in range(total_chunk_num):
    # 获取当前分块的音频数据
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    
    # 检查当前分块是否为最后一个分块
    is_final = i == total_chunk_num - 1
    
    # 生成模型的输出，传入当前分块音频数据和其他参数
    res = model.generate(
        input=speech_chunk, 
        cache=cache, 
        is_final=is_final, 
        chunk_size=chunk_size, 
        encoder_chunk_look_back=encoder_chunk_look_back, 
        decoder_chunk_look_back=decoder_chunk_look_back
    )
    
    # 打印当前分块的生成结果
    print(res)