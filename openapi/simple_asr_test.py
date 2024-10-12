"""
Description: `example_wav/asr_example.wav` 单次测试示例。
Notes: 
每次测试会输出耗时。
"""
import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本的父目录的父目录
parent_directory_of_the_parent_directory = os.path.dirname(os.path.dirname(current_script_path))
# 将这个目录添加到 sys.path
sys.path.append(parent_directory_of_the_parent_directory)

import aiohttp
import asyncio
import time

async def test_asr(api_url, audio_file_path):
    async with aiohttp.ClientSession() as session:
        with open(audio_file_path, 'rb') as audio_file:
            data = aiohttp.FormData()
            data.add_field('file', audio_file, filename=audio_file_path.split('/')[-1], content_type='audio/wav')
            async with session.post(api_url, data=data) as response:
                return await response.json()

async def main():
    api_url = "http://localhost:8847/simple_asr"  # 根据你的实际地址进行修改
    audio_file_path = "example_wav/asr_example.wav"  # 替换为你的 WAV 文件路径
    
    start_time = time.perf_counter()  # 记录开始时间
    result = await test_asr(api_url, audio_file_path)
    end_time = time.perf_counter()  # 记录结束时间
    
    print(f"ASR Result: {result}")
    print(f"Total Time Taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
