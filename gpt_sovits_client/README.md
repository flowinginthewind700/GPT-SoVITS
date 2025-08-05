# GPT-SoVITS Client SDK

GPT-SoVITS Client SDK 是一个用于多语言混合文本转语音(TTS)的Python客户端库。

## 🌟 特性

- **多语言支持**: 支持中文、英文、日文、韩文等多种语言
- **混合语言处理**: 自动检测和分割混合语言文本
- **简单易用**: 提供简洁的API接口
- **类型安全**: 完整的类型注解支持
- **错误处理**: 完善的异常处理机制
- **批量处理**: 支持批量TTS合成

## 📦 安装

### 从PyPI安装

```bash
pip install gpt-sovits-client
```

### 从源码安装

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS/gpt_sovits_client
pip install -e .
```

## 🚀 快速开始

### 基本使用

```python
from gpt_sovits_client import GPTSoVITSClient, LanguageType

# 初始化客户端
client = GPTSoVITSClient(base_url="http://localhost:9880")

# 简单TTS合成
response = client.synthesize_text(
    text="Hello 你好 world 世界",
    ref_audio_path="sample.wav",
    text_lang=LanguageType.AUTO  # 自动检测语言
)

if response.success:
    print(f"✅ 生成成功: {response.audio_path}")
    print(f"📊 文件大小: {response.file_size/1024:.1f}KB")
else:
    print(f"❌ 生成失败: {response.message}")
```

### 高级使用

```python
from gpt_sovits_client import TTSRequest, LanguageType, TextSplitMethod

# 创建详细的请求对象
request = TTSRequest(
    text="这是一个详细的TTS请求示例",
    ref_audio_path="sample.wav",
    text_lang=LanguageType.ALL_ZH,
    text_split_method=TextSplitMethod.CUT5,
    top_k=5,
    top_p=1.0,
    temperature=1.0,
    speed_factor=1.0,
    repetition_penalty=1.35
)

# 执行合成
response = client.synthesize(request)
```

## 📚 API 文档

### GPTSoVITSClient

主要的客户端类。

#### 初始化

```python
client = GPTSoVITSClient(
    base_url="http://localhost:9880",  # API服务地址
    timeout=180  # 请求超时时间（秒）
)
```

#### 方法

##### `health_check() -> bool`

检查API服务健康状态。

##### `synthesize_text(text, ref_audio_path, **kwargs) -> TTSResponse`

简化版TTS合成。

**参数:**
- `text` (str): 要合成的文本
- `ref_audio_path` (str): 参考音频文件路径
- `text_lang` (LanguageType): 文本语言类型
- `prompt_text` (str): 提示文本
- `prompt_lang` (LanguageType): 提示语言类型
- `**kwargs`: 其他参数

##### `synthesize(request) -> TTSResponse`

使用TTSRequest对象进行TTS合成。

##### `batch_synthesize(requests) -> List[TTSResponse]`

批量TTS合成。

##### `detect_language_segments(text) -> List[Dict[str, str]]`

检测文本中的语言片段。

##### `is_mixed_language(text) -> bool`

检查是否为混合语言文本。

##### `get_primary_language(text) -> str`

获取文本的主要语言。

##### `auto_detect_language(text) -> LanguageType`

自动检测语言类型。

### LanguageType

支持的语言类型枚举：

- `AUTO`: 自动检测
- `ALL_ZH`: 全部按中文处理
- `EN`: 英文
- `JA`: 日文
- `KO`: 韩文
- `YUE`: 粤语
- `ALL_JA`: 全部按日文处理
- `ALL_KO`: 全部按韩文处理
- `ALL_YUE`: 全部按粤语处理

### TextSplitMethod

文本分割方法枚举：

- `CUT0`: 按标点符号分割
- `CUT1`: 按句子分割
- `CUT2`: 按字符数分割
- `CUT3`: 按语义分割
- `CUT4`: 按段落分割
- `CUT5`: 智能分割（推荐）

### TTSRequest

TTS请求参数类。

### TTSResponse

TTS响应结果类。

**属性:**
- `success` (bool): 是否成功
- `audio_path` (str): 音频文件路径
- `file_size` (int): 文件大小
- `message` (str): 消息
- `error_code` (int): 错误代码

## 🌍 多语言支持

### 语言检测

SDK内置智能语言检测功能：

```python
# 检测语言片段
segments = client.detect_language_segments("Hello 你好 world 世界")
# 结果: [{"text": "Hello", "lang": "en"}, {"text": "你好", "lang": "zh"}, ...]

# 检查混合语言
is_mixed = client.is_mixed_language("Hello 你好 world 世界")
# 结果: True

# 获取主要语言
primary_lang = client.get_primary_language("Hello 你好 world 世界")
# 结果: "en"
```

### 支持的语言组合

- **中英混合**: `Hello 你好 world 世界`
- **中日混合**: `こんにちは 你好 世界`
- **中韩混合**: `안녕하세요 你好 世界`
- **英日混合**: `Hello こんにちは world`
- **复杂混合**: `Hello こんにちは 你好 안녕하세요 world 世界`

## 🔧 配置

### 环境变量

- `GPT_SOVITS_API_URL`: API服务地址（默认: http://localhost:9880）
- `GPT_SOVITS_TIMEOUT`: 请求超时时间（默认: 180秒）

### 配置文件

可以通过配置文件自定义设置：

```python
import os
os.environ["GPT_SOVITS_API_URL"] = "http://your-api-server:9880"
os.environ["GPT_SOVITS_TIMEOUT"] = "300"
```

## 🐳 Docker 部署

### 服务端部署

```bash
# 构建镜像
docker build -t gpt-sovits-api .

# 运行容器
docker run -d \
  --name gpt-sovits-api \
  -p 9880:9880 \
  -v ./models:/app/models \
  -v ./logs:/app/logs \
  gpt-sovits-api
```

### 使用Docker Compose

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f gpt-sovits-api

# 停止服务
docker-compose down
```

## 📝 示例

### 基本示例

```python
from gpt_sovits_client import GPTSoVITSClient, LanguageType

client = GPTSoVITSClient()

# 中文TTS
response = client.synthesize_text(
    text="你好，这是一个中文语音合成测试。",
    ref_audio_path="sample.wav",
    text_lang=LanguageType.ALL_ZH
)

# 英文TTS
response = client.synthesize_text(
    text="Hello, this is an English TTS test.",
    ref_audio_path="sample.wav",
    text_lang=LanguageType.EN
)

# 混合语言TTS
response = client.synthesize_text(
    text="Hello 你好 world 世界",
    ref_audio_path="sample.wav",
    text_lang=LanguageType.AUTO
)
```

### 高级示例

```python
from gpt_sovits_client import TTSRequest, LanguageType, TextSplitMethod

# 创建请求对象
request = TTSRequest(
    text="这是一个高级TTS示例",
    ref_audio_path="sample.wav",
    text_lang=LanguageType.ALL_ZH,
    text_split_method=TextSplitMethod.CUT5,
    top_k=5,
    top_p=1.0,
    temperature=1.0,
    speed_factor=1.0,
    repetition_penalty=1.35
)

# 执行合成
response = client.synthesize(request)
```

## 🛠️ 开发

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black gpt_sovits_client/
flake8 gpt_sovits_client/
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 支持

- GitHub Issues: https://github.com/RVC-Boss/GPT-SoVITS/issues
- 文档: https://github.com/RVC-Boss/GPT-SoVITS/wiki 