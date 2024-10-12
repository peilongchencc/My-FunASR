"""
Description: `example_wav/asr_example.wav` 多次测试示例。
Notes: 
每次测试会输出耗时，全部结束后会输出平均耗时。
"""
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

async def run_test(api_url, audio_file_path, run_id):
    start_time = time.perf_counter()  # 记录开始时间
    result = await test_asr(api_url, audio_file_path)
    end_time = time.perf_counter()  # 记录结束时间

    total_time = end_time - start_time
    print(f"Run {run_id} - ASR Result: {result}")
    print(f"Run {run_id} - Total Time Taken: {total_time:.2f} seconds")
    return total_time

async def main():
    api_url = "http://localhost:8847/simple_asr"  # 根据你的实际地址进行修改
    audio_file_path = "example_wav/asr_example.wav"  # 替换为你的 WAV 文件路径

    num_runs = 20  # 设置运行次数
    total_times = []  # 用于保存每次运行的时间

    # 按照设置的次数运行 ASR 测试
    for run_id in range(1, num_runs + 1):
        total_time = await run_test(api_url, audio_file_path, run_id)
        total_times.append(total_time)

    # 计算平均时间
    average_time = sum(total_times) / len(total_times)
    print(f"\nAverage Time Taken for {num_runs} runs: {average_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())