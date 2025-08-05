# TTS API with Voice Management

这是一个增强版的GPT-SoVITS TTS API，支持voice管理功能。

## 功能特性

- ✅ 支持voice管理（预定义的音色）
- ✅ 自动处理参考音频和提示文本
- ✅ 简单的HTTP API接口
- ✅ Python客户端示例
- ✅ 支持性别标记（male/female）

## 目录结构

```
GPT-SoVITS/
├── tts_api.py              # 增强版TTS API服务器
├── tts_client.py           # Python客户端
├── example_usage.py        # 使用示例
├── voice/                  # Voice目录
│   ├── voice1/            # 女声voice
│   │   ├── config.json    # 配置文件
│   │   ├── sample.wav     # 音频文件（需要手动添加）
│   │   └── sample.wav.txt # 对应的文本
│   ├── voice2/            # 男声voice
│   │   ├── config.json
│   │   ├── sample.wav
│   │   └── sample.wav.txt
│   └── voice3/            # 其他voice
└── README_TTS_API.md      # 本文档
```

## 启动API服务器

```bash
# 启动TTS API服务器
python tts_api.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

## Voice管理

### 1. Voice目录结构

每个voice目录包含：
- `config.json`: 配置文件（性别、描述等）
- `sample.wav`: 参考音频文件
- `sample.wav.txt`: 对应的文本内容

### 2. 配置文件格式

```json
{
    "name": "voice1",
    "gender": "female",
    "description": "温柔女声",
    "language": "zh"
}
```

### 3. 添加新的Voice

1. 在`voice/`目录下创建新目录（如`voice4/`）
2. 添加音频文件（如`sample.wav`）
3. 添加对应的文本文件（如`sample.wav.txt`）
4. 创建配置文件`config.json`

## API接口

### 1. 获取Voice列表

```bash
GET /voices
```

响应：
```json
{
    "voices": {
        "voice1": {
            "name": "voice1",
            "gender": "female",
            "description": "温柔女声",
            "audio_files": ["sample.wav"],
            "text_files": ["sample.wav.txt"]
        }
    }
}
```

### 2. 获取Voice详情

```bash
GET /voice/{voice_name}
```

### 3. 设置Voice

```bash
POST /voice/{voice_name}/set?audio_file=sample.wav
```

### 4. 使用Voice进行TTS（推荐）

```bash
POST /tts_with_voice
```

参数：
- `voice_name`: voice名称
- `text`: 要合成的文本
- `text_lang`: 文本语言（默认zh）
- `audio_file`: 指定音频文件（可选）
- 其他TTS参数...

## Python客户端使用

### 1. 基本使用

```python
from tts_client import TTSClient

# 创建客户端
client = TTSClient("http://127.0.0.1:9880")

# 检查API状态
if not client.check_health():
    print("API服务未运行")
    exit()

# 获取voice列表
voices = client.list_voices()
print(f"可用voice: {list(voices['voices'].keys())}")

# 使用voice进行TTS
audio_data = client.tts_with_voice(
    voice_name="voice1",
    text="你好，这是一个测试！",
    output_file="output.wav"
)
```

### 2. 运行示例

```bash
# 运行完整示例
python tts_client.py

# 运行简单示例
python example_usage.py
```

## 使用步骤

1. **启动API服务器**
   ```bash
   python tts_api.py
   ```

2. **准备Voice文件**
   - 将音频文件放入`voice/voice1/sample.wav`
   - 创建对应的文本文件`voice/voice1/sample.wav.txt`

3. **使用客户端**
   ```bash
   python example_usage.py
   ```

## 注意事项

1. **音频文件格式**: 支持wav、mp3、flac、m4a格式
2. **文本编码**: 文本文件使用UTF-8编码
3. **文件命名**: 文本文件应与音频文件同名，扩展名为`.txt`
4. **API端口**: 默认使用9880端口，可通过参数修改

## 故障排除

1. **API服务未启动**
   - 检查是否运行`python tts_api.py`
   - 检查端口是否被占用

2. **Voice未找到**
   - 检查`voice/`目录结构
   - 确认配置文件格式正确

3. **音频生成失败**
   - 检查参考音频文件是否存在
   - 确认音频文件格式正确
   - 查看API日志获取详细错误信息 