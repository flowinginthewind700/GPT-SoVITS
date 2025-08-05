# 🎙️ TTS API 使用指南

## 🌐 服务器信息
- **地址**: http://219.144.21.182:9880
- **协议**: HTTP/HTTPS
- **支持**: 多语言混合推理 (text_lang=auto)
- **优化**: CUDA加速 + 智能分块处理

## 📋 可用音色
- **vivienne** (女声) - 英文参考音频
- **allen** (男声) - 中文参考音频

## 🔧 主要API端点

### 1. 标准TTS接口 (推荐)

```http
POST /tts
Content-Type: application/json
```

**请求参数:**
```json
{
  "text": "你好world，这是一个test！",           // 必需：要合成的文本
  "text_lang": "auto",                        // 必需：auto=自动检测多语言
  "ref_audio_path": "voice/vivienne/sample.mp3", // 必需：参考音频路径
  "prompt_text": "Hello, this is a sample text.", // 可选：参考音频的文本
  "prompt_lang": "en",                        // 必需：参考音频语言
  "temperature": 1.0,                         // 可选：温度参数 (0.1-2.0)
  "speed_factor": 1.0,                        // 可选：语速控制 (0.5-2.0)
  "sample_steps": 32,                         // 可选：采样步数
  "top_k": 5,                                 // 可选：top-k采样
  "top_p": 1.0,                              // 可选：top-p采样
  "text_split_method": "cut5",               // 可选：文本分割方法
  "batch_size": 1,                           // 可选：批处理大小
  "media_type": "wav",                       // 可选：输出格式
  "streaming_mode": false                    // 可选：流式输出
}
```

### 2. 缓存音色接口 (更快)

```http
POST /tts_with_cached_voice
Content-Type: application/x-www-form-urlencoded
```

**请求参数:**
```
voice_name=vivienne
text=你好world，这是一个test！
text_lang=auto
temperature=1.0
speed_factor=1.0
sample_steps=32
```

### 3. 性能监控接口

```http
GET /performance_stats         # 获取性能统计
GET /optimization_status       # 获取优化状态
POST /toggle_optimization      # 控制优化开关
```

## 💻 代码示例

### Python 示例

```python
import requests

# 方法1: 标准TTS API
def call_tts_standard(text, voice="vivienne"):
    url = "http://219.144.21.182:9880/tts"
    payload = {
        "text": text,
        "text_lang": "auto",  # 自动检测多语言
        "ref_audio_path": f"voice/{voice}/sample.mp3",
        "prompt_text": "Hello, this is a sample text.",
        "prompt_lang": "en" if voice == "vivienne" else "zh",
        "temperature": 1.0,
        "speed_factor": 1.0,
        "sample_steps": 32
    }
    
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code == 200:
        with open("output.wav", "wb") as f:
            f.write(response.content)
        return True
    return False

# 方法2: 缓存音色API (更快)
def call_tts_cached(text, voice="vivienne"):
    url = "http://219.144.21.182:9880/tts_with_cached_voice"
    params = {
        "voice_name": voice,
        "text": text,
        "text_lang": "auto",
        "temperature": 1.0,
        "speed_factor": 1.0
    }
    
    response = requests.post(url, params=params, timeout=120)
    if response.status_code == 200:
        with open("output_cached.wav", "wb") as f:
            f.write(response.content)
        return True
    return False

# 使用示例
texts = [
    "你好world，这是一个test！",
    "Hello世界，多语言混合测试です。",
    "This is pure English text.",
    "这是纯中文文本。"
]

for i, text in enumerate(texts):
    print(f"生成第{i+1}个音频...")
    success = call_tts_cached(text, "vivienne")
    if success:
        print(f"✅ 成功生成: output_cached.wav")
    else:
        print("❌ 生成失败")
```

### JavaScript/Fetch 示例

```javascript
async function generateSpeech(text, voice = 'vivienne') {
    const url = 'http://219.144.21.182:9880/tts';
    const payload = {
        text: text,
        text_lang: 'auto',  // 自动检测多语言
        ref_audio_path: `voice/${voice}/sample.mp3`,
        prompt_text: 'Hello, this is a sample text.',
        prompt_lang: voice === 'allen' ? 'zh' : 'en',
        temperature: 1.0,
        speed_factor: 1.0,
        sample_steps: 32,
        media_type: 'wav'
    };
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // 创建音频元素播放
            const audio = new Audio(audioUrl);
            audio.play();
            
            return true;
        } else {
            console.error('TTS failed:', response.status);
            return false;
        }
    } catch (error) {
        console.error('Request failed:', error);
        return false;
    }
}

// 使用示例
generateSpeech('你好world，这是JavaScript调用示例！', 'vivienne');
```

### cURL 示例

```bash
# 标准TTS API
curl -X POST "http://219.144.21.182:9880/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好world，这是cURL调用示例！",
    "text_lang": "auto",
    "ref_audio_path": "voice/vivienne/sample.mp3",
    "prompt_text": "Hello, this is a sample text.",
    "prompt_lang": "en",
    "temperature": 1.0,
    "speed_factor": 1.0,
    "sample_steps": 32
  }' \
  --output "output.wav"

# 缓存音色API (更快)
curl -X POST "http://219.144.21.182:9880/tts_with_cached_voice" \
  -d "voice_name=vivienne" \
  -d "text=你好world，这是缓存音色调用！" \
  -d "text_lang=auto" \
  -d "temperature=1.0" \
  -d "speed_factor=1.0" \
  --output "output_cached.wav"
```

## 🌍 多语言支持

### text_lang 参数说明

- **auto** (推荐): 自动检测并支持多语言混合
- **zh**: 中文
- **en**: 英文  
- **ja**: 日文
- **ko**: 韩文

### 多语言混合示例

```python
# 支持的多语言混合文本
multilingual_texts = [
    "你好world，这是一个test！",                    # 中英混合
    "Hello世界，今天の天気はgood！",                # 英中日混合  
    "안녕하세요，你好，Hello everybody！",          # 韩中英混合
    "これはtest，包含Japanese和中文content。",      # 日英中混合
]

for text in multilingual_texts:
    call_tts_cached(text, voice="vivienne")
```

## ⚡ 性能优化特性

### CUDA优化状态
```python
# 检查优化状态
response = requests.get("http://219.144.21.182:9880/optimization_status")
status = response.json()

print(f"CUDA可用: {status['cuda_available']}")
print(f"优化启用: {status['optimization_enabled']}")
print(f"配置: {status['config']}")
```

### 性能监控
```python
# 获取性能统计
response = requests.get("http://219.144.21.182:9880/performance_stats")
stats = response.json()

print(f"总请求数: {stats['total_requests']}")
print(f"平均处理时间: {stats['avg_tts_time']:.3f}s")
print(f"吞吐量: {stats['requests_per_second']:.3f} 请求/s")
```

## 🎛️ 参数调优建议

### 温度参数 (temperature)
- **0.1-0.5**: 稳定，一致性高
- **0.8-1.2**: 平衡 (推荐)
- **1.5-2.0**: 多样性高，更有表现力

### 语速控制 (speed_factor)
- **0.5-0.8**: 慢速
- **0.9-1.1**: 正常 (推荐)
- **1.2-2.0**: 快速

### 采样步数 (sample_steps)
- **16-24**: 快速，质量一般
- **32**: 平衡 (推荐)
- **48-64**: 高质量，较慢

## 🔧 故障排除

### 常见错误

1. **连接超时**
   - 检查网络连接
   - 增加timeout时间

2. **音色不存在**
   - 使用 `GET /voices` 获取可用音色
   - 确认音色名称正确

3. **文本过长**
   - 建议单次请求不超过200字符
   - 使用文本分割

4. **优化未启用**
   - 使用 `POST /toggle_optimization?enable=true` 启用

### 性能建议

1. **使用缓存音色API** - 更快的响应
2. **启用CUDA优化** - 显著性能提升
3. **合理设置参数** - 平衡质量和速度
4. **批量处理** - 减少网络开销

## 📊 预期性能

| 文本长度 | 处理时间 | 处理速度 |
|---------|---------|---------|
| 短文本 (10字符) | 1-2秒 | 5-10 字符/s |
| 中等文本 (30字符) | 2-4秒 | 8-15 字符/s |
| 长文本 (50字符) | 4-8秒 | 6-12 字符/s |

*注：实际性能取决于服务器负载和网络状况*

## 🎉 快速开始

1. **下载工具**: [simple_tts_client.py](./simple_tts_client.py)
2. **基础调用**:
   ```bash
   python simple_tts_client.py "你好world，这是测试！" -v vivienne -l auto
   ```
3. **网页界面**: 打开 [web_api_example.html](./web_api_example.html)
4. **检查状态**:
   ```bash
   python simple_tts_client.py --status
   ```

享受高质量的多语言TTS服务！🎤✨