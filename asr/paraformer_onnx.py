import time
from funasr_onnx import SeacoParaformer

# paraformer-zh 是一个多功能的自动语音识别（ASR）模型，可以根据需要选择使用语音活动检测（VAD）、标点符号（punc）和说话人识别（spk）。
class ALIASR():
    def __init__(self, model_dir = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"):
        # 默认gpu0 推理
        self.model = SeacoParaformer(model_dir, batch_size=1, device_id=0)
        
    def transcription(self, audio, hotwords=""):
        start_time = time.time()
        res = self.model(audio, hotwords)
        # 结束计时
        end_time = time.time()

        # 计算并输出预测时间
        print(f"模型预测时间：{end_time - start_time:.4f} 秒")
        
        res_sentence = "".join(res[0]['raw_tokens'])
        print(f"funasr预测结果:{res_sentence}")
        return res_sentence