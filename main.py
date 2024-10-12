"""
Description: 
Notes: 
Requirements:
pip install python-multipart
"""
import json
from loguru import logger
from asr.paraformer import ALIASR
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

# 根目录访问的处理
@app.get("/")
async def read_root():
    return json.dumps({"code": 0, "msg": "欢迎访问ASR", "data": ""})

@app.post("/simple_asr")
async def simple_asr(file: UploadFile = File(...)):
    """语音转文字"""
    try:
        # 读取文件内容为字节流, audio_bytes文件类型为:<class 'bytes'>
        audio = await file.read()
        recognized_text = auto_asr.transcription(audio)
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
