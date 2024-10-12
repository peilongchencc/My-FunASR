"""
Description: wav文件采样率转换示例。
Notes: 
pydub依赖与ffmpeg，安装方式: `apt install ffmpeg`
"""
from pydub import AudioSegment
import os

def convert_sample_rate(input_file, output_file, new_sample_rate=16000):
    try:
        # 加载音频文件
        audio = AudioSegment.from_wav(input_file)
        
        # 转换采样率
        audio = audio.set_frame_rate(new_sample_rate)
        
        # 导出转换后的音频文件
        audio.export(output_file, format='wav')
        print(f"成功将音频转换为 {new_sample_rate} Hz，文件保存在 {output_file}。")
        
    except FileNotFoundError:
        print(f"输入文件 {input_file} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    input_path = "input_8000.wav"
    output_path = "output_16000.wav"
    
    # 检查输出目录是否存在
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    convert_sample_rate(input_path, output_path)