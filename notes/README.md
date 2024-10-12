# 项目相关

🚨注意: 如果大家想要了解 FunASR 的使用，请以 FunASR 的 GitHub 为准，Model Scope 不为标准。

- [项目相关](#项目相关)
  - [官方示例:](#官方示例)
    - [非实时语音识别:](#非实时语音识别)
  - [热词的作用:](#热词的作用)
  - [附录--FFmpeg安装:](#附录--ffmpeg安装)
  - [附录--分析和获取媒体文件的信息:](#附录--分析和获取媒体文件的信息)
    - [FFmpeg和FFprobe的关系？](#ffmpeg和ffprobe的关系)
  - [附录--git下载模型(可选):](#附录--git下载模型可选)


## 官方示例:

利用FunASR框架加载paraformer模型。(类似VLLM加载ChatGLM模型)

### 非实时语音识别:

```python
from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh",  
                  vad_model="fsmn-vad", 
                  vad_kwargs={"max_single_segment_time": 60000},
                  punc_model="ct-punc", 
                  # spk_model="cam++"
                  )
wav_file = f"{model.model_path}/example/asr_example.wav"
res = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
print(res)
```

下载的模型默认保存在 `~/.cache/modelscope/hub/` 目录。

注意：

- 通常模型输入限制时长30s以下，组合`vad_model`后，支持任意时长音频输入，不局限于paraformer模型，所有音频输入模型均可以。
- `model`相关的参数可以直接在`AutoModel`定义中直接指定；与`vad_model`相关参数可以通过`vad_kwargs`来指定，类型为dict；类似的有`punc_kwargs`，`spk_kwargs`；
- `max_single_segment_time`: 表示`vad_model`最大切割音频时长, 单位是毫秒ms.
- `batch_size_s` 表示采用动态batch，batch中总音频时长，单位为秒s。
- `batch_size_threshold_s`: 表示`vad_model`切割后音频片段时长超过 `batch_size_threshold_s`阈值时，将batch_size数设置为1, 单位为秒s.

建议：当您输入为长音频，遇到OOM问题时，因为显存占用与音频时长呈平方关系增加，分为3种情况：

- a)推理起始阶段，显存主要取决于`batch_size_s`，适当减小该值，可以减少显存占用；
- b)推理中间阶段，遇到VAD切割的长音频片段，总token数小于`batch_size_s`，仍然出现OOM，可以适当减小`batch_size_threshold_s`，超过阈值，强制batch为1; 
- c)推理快结束阶段，遇到VAD切割的长音频片段，总token数小于`batch_size_s`，且超过阈值`batch_size_threshold_s`，强制batch为1，仍然出现OOM，可以适当减小`max_single_segment_time`，使得VAD切割音频时长变短。


## 热词的作用:

以 "会议" 为例，大多数情况下，语言模型会将 "我今天参加了一个会议" 的声音都会识别为 "会议"。

但有时，例如 "老师讲课时，已经讲过'会意'的含义了？" 这句话，paraformer 可能会将 "会意" 识别为 "会议"，此时自己就添加下:

```python
hotwords = "会意"
```

有时公司名，组织名也会用到热词，多个热词以空格作为分隔是吧。


## 附录--FFmpeg安装:

Linux安装FFmpeg指令:

```bash
sudo apt update
sudo apt install ffmpeg
```

FFmpeg版本查看:

```bash
ffmpeg -version
```


## 附录--分析和获取媒体文件的信息:

假设你要分析 `example_wav/asr_example.wav` 文件的媒体信息:

```bash
ffprobe example_wav/asr_example.wav
```

终端输出:

```log
(my_env) root@ubuntu22:/data/paraformer# ffprobe example_wav/asr_example.wav
ffprobe version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2007-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Input #0, wav, from 'example_wav/asr_example.wav':
  Metadata:
    encoder         : Lavf57.82.104
  Duration: 00:00:05.55, bitrate: 256 kb/s
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, 1 channels, s16, 256 kb/s
(my_env) root@ubuntu22:/data/paraformer# 
```

### FFmpeg和FFprobe的关系？

FFmpeg 和 FFprobe 都是 FFmpeg 项目的一部分，但它们的功能不同：

1. **FFmpeg**：用于录制、转换和流式传输音频和视频。它可以处理多种格式，进行格式转换、剪辑、合并、添加效果等。

2. **FFprobe**：用于分析和获取媒体文件的信息。它可以提取视频、音频流的详细元数据，例如编码格式、分辨率、时长、比特率等，通常用于调试或获取文件特性。

简单来说，FFmpeg 主要用于处理多媒体文件，而 FFprobe 主要用于分析这些文件。


## 附录--git下载模型(可选):

```bash
git lfs install
git clone https://www.modelscope.cn/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git
```