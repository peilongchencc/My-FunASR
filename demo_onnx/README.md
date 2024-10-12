# funasr_onnx 基础示例

- [funasr\_onnx 基础示例](#funasr_onnx-基础示例)
  - [文件简介:](#文件简介)
  - [onnx模型使用注意事项:](#onnx模型使用注意事项)


## 文件简介:

| 文件名                          | 作用                                    | 备注                                         |
|--------------------------------|----------------------------------------|----------------------------------------------|
| demo_seaco_export.py           | 混合模型转onnx格式示例                    | 转换出的模型为demo_seaco_paraformer*使用的模型🚨 |
| export_paraformer_onnx.py      | 单独模型转onnx格式示例                    |                                              |
| load_onnx_model.py             | ASR onnx模型调用示例                     |                                              |
| demo_paraformer_unonnx.py      | funasr ASR、VAD、PUNC模型组合版运行基础示例 |                                              |
| demo_seaco_paraformer.py       | funasr_onnx模型组合版运行基础示例          | 输入为wav文件                                  |
| demo_seaco_paraformer_pydub.py | funasr_onnx模型组合版运行bytes转np示例     | onnx模型不接受bytes类型数据                      |


## onnx模型使用注意事项:

混合版onnx模型、ASR onnx模型、VAD onnx模型的调用方式均不同，注意区分。详情可查看[FunASR onnx专栏](https://github.com/modelscope/FunASR/tree/main/runtime/python/onnxruntime)。
