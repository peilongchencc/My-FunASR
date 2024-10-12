from funasr import AutoModel
# paraformer-zh 是一个多功能的自动语音识别（ASR）模型，可以根据需要选择使用语音活动检测（VAD）、标点符号（punc）和说话人识别（spk）。
class ALIASR():
    def __init__(self, asr_model_path: str = "paraformer-zh", 
                 vad_model_path: str = "fsmn-vad", punc_model_path: str = 'ct-punc'):
        # 默认gpu0 推理
        self.model = AutoModel(model=asr_model_path,
                        vad_model=vad_model_path,
                        punc_model=punc_model_path,
                        disable_update=True,    # 不提示更新funasr
                        )
        
    def transcription(self, audio):
        res = self.model.generate(input=audio, max_end_silence_time=2000,
                            batch_size_s=10)
        print(f"funasr预测结果:{res[0]['text']}")
        return res[0]['text']