"""
Description: ASR主程序，调用onnx格式的模型。(比常规模型反应速度更快)
Notes: 
Requirements:
pip install loguru pydub python-multipart
"""
import io
import json
import numpy as np
from loguru import logger
from pydub import AudioSegment
from asr.paraformer_onnx import ALIASR
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager

auto_asr: ALIASR = None  # 全局变量

@asynccontextmanager
async def lifespan(app: FastAPI):
    """程序启动前加载模型"""
    global auto_asr
    auto_asr = ALIASR()
    yield
    """销毁模型"""
    auto_asr = None

app = FastAPI(lifespan=lifespan)

# 定义转换函数：将二进制音频数据转换为 numpy 数组格式
def bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    # 使用 io.BytesIO 将字节流转换为文件流对象
    audio_stream = io.BytesIO(audio_bytes)
    # 使用 pydub 将文件流对象读取为 AudioSegment
    audio = AudioSegment.from_file(audio_stream, format="wav")
    # 将音频数据转换为 numpy 数组，并归一化到 [-1.0, 1.0] 范围
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return audio_data

# 根目录访问的处理
@app.get("/")
async def read_root():
    return json.dumps({"code": 0, "msg": "欢迎访问ASR", "data": ""})

@app.post("/simple_asr")
async def simple_asr(file: UploadFile = File(...)):
    """语音转文字"""
    try:
        # 读取文件内容为字节流, audio_bytes文件类型为:<class 'bytes'>
        audio_bytes = await file.read()
        # 将字节流转换为 numpy 数组格式, audio_np文件类型为:<class 'numpy.ndarray'>
        audio_np = bytes_to_numpy(audio_bytes)
        recognized_text = auto_asr.transcription(audio_np)
        return JSONResponse(
            content={"code": 0, "msg": "ASR识别成功", "data": {"recognized_text": recognized_text}},
            status_code=200
        )
    except Exception as e:
        logger.error(f"错误信息为: {str(e)}")
        return JSONResponse(
            content={"code": 1, "msg": "识别失败", "data": {"recognized_text": ''}},
            status_code=400
        )

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8847)
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}")
