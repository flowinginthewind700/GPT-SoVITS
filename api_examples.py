#!/usr/bin/env python3
"""
TTS API 调用示例 - 支持多语言混合推理
服务器地址: http://219.144.21.182:9880
"""
import requests
import json
from typing import Optional

# 服务器配置
BASE_URL = "http://219.144.21.182:9880"

def call_tts_api(
    text: str,
    voice_name: str = "vivienne",
    text_lang: str = "auto",  # 使用auto进行多语言自动检测
    prompt_lang: str = "zh",  # 参考音频的语言
    temperature: float = 1.0,
    speed_factor: float = 1.0,
    sample_steps: int = 32,
    save_path: Optional[str] = None
):
    """
    调用TTS API生成语音
    
    Args:
        text: 要合成的文本（支持多语言混合）
        voice_name: 音色名称（vivienne, allen等）
        text_lang: 文本语言（auto=自动检测多语言）
        prompt_lang: 参考音频语言
        temperature: 温度参数，控制随机性
        speed_factor: 语速控制
        sample_steps: 采样步数
        save_path: 保存路径（可选）
    
    Returns:
        tuple: (success: bool, result: bytes/str)
    """
    
    # 构建请求数据
    payload = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": f"voice/{voice_name}/sample.mp3",
        "prompt_text": get_prompt_text(voice_name),
        "prompt_lang": prompt_lang,
        "temperature": temperature,
        "speed_factor": speed_factor,
        "sample_steps": sample_steps,
        "top_k": 5,
        "top_p": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False
    }
    
    try:
        print(f"🎯 生成语音: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"🎨 使用音色: {voice_name}")
        print(f"🌍 语言模式: {text_lang}")
        
        # 发送POST请求
        response = requests.post(
            f"{BASE_URL}/tts",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            # 成功获取音频数据
            audio_data = response.content
            
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(audio_data)
                print(f"✅ 音频已保存到: {save_path}")
            
            return True, audio_data
        else:
            error_msg = f"API调用失败 - 状态码: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f", 错误: {error_detail.get('message', 'Unknown error')}"
            except:
                error_msg += f", 响应: {response.text[:100]}"
            
            print(f"❌ {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"请求异常: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

def get_prompt_text(voice_name: str) -> str:
    """获取音色对应的提示文本"""
    prompts = {
        "vivienne": "Hello, this is a sample text.",
        "allen": "你好，这是一个示例文本。",
        # 可以根据需要添加更多音色的提示文本
    }
    return prompts.get(voice_name, "Hello, this is a sample text.")

def call_cached_voice_api(
    text: str,
    voice_name: str = "vivienne",
    text_lang: str = "auto",
    temperature: float = 1.0,
    speed_factor: float = 1.0,
    save_path: Optional[str] = None
):
    """
    使用缓存音色API（更快的响应）
    """
    try:
        print(f"🚀 使用缓存音色生成: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"🎨 音色: {voice_name}")
        
        # 使用缓存音色接口
        params = {
            "voice_name": voice_name,
            "text": text,
            "text_lang": text_lang,
            "temperature": temperature,
            "speed_factor": speed_factor,
            "sample_steps": 32
        }
        
        response = requests.post(
            f"{BASE_URL}/tts_with_cached_voice",
            params=params,
            timeout=120
        )
        
        if response.status_code == 200:
            audio_data = response.content
            
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(audio_data)
                print(f"✅ 音频已保存到: {save_path}")
            
            return True, audio_data
        else:
            error_msg = f"缓存音色API调用失败 - 状态码: {response.status_code}"
            print(f"❌ {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"请求异常: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

def get_available_voices():
    """获取可用的音色列表"""
    try:
        response = requests.get(f"{BASE_URL}/voices", timeout=10)
        if response.status_code == 200:
            voices_data = response.json()
            return voices_data.get("voices", {})
        else:
            print(f"❌ 获取音色列表失败: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ 获取音色列表异常: {e}")
        return {}

def main():
    """示例演示"""
    print("🎙️ TTS API 调用示例")
    print("=" * 50)
    
    # 1. 获取可用音色
    print("📋 获取可用音色...")
    voices = get_available_voices()
    if voices:
        print("✅ 可用音色:")
        for voice_name, voice_info in voices.items():
            print(f"   - {voice_name}: {voice_info.get('description', 'No description')}")
    else:
        print("⚠️ 无法获取音色列表，使用默认音色")
    
    # 2. 多语言混合文本示例
    test_cases = [
        {
            "name": "中英混合",
            "text": "你好world，这是一个test，包含中文和English mixed content。",
            "voice": "vivienne"
        },
        {
            "name": "纯中文",
            "text": "这是一个纯中文的语音合成测试。",
            "voice": "allen"
        },
        {
            "name": "纯英文",
            "text": "This is a pure English text-to-speech synthesis test.",
            "voice": "vivienne"
        },
        {
            "name": "中日英混合",
            "text": "你好，こんにちは，Hello world！多语言混合测试です。",
            "voice": "vivienne"
        }
    ]
    
    # 3. 执行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 测试 {i}: {test_case['name']}")
        print("-" * 30)
        
        # 方法1: 使用标准TTS API
        success, result = call_tts_api(
            text=test_case["text"],
            voice_name=test_case["voice"],
            text_lang="auto",  # 自动检测语言
            save_path=f"output_{i}_standard.wav"
        )
        
        if success:
            print(f"✅ 标准API调用成功，音频大小: {len(result)/1024:.1f}KB")
        
        # 方法2: 使用缓存音色API（通常更快）
        success, result = call_cached_voice_api(
            text=test_case["text"],
            voice_name=test_case["voice"],
            text_lang="auto",
            save_path=f"output_{i}_cached.wav"
        )
        
        if success:
            print(f"✅ 缓存API调用成功，音频大小: {len(result)/1024:.1f}KB")

# HTTP请求示例（curl格式）
def print_curl_examples():
    """打印curl命令示例"""
    print("\n📡 cURL 命令示例:")
    print("=" * 50)
    
    # 标准TTS API
    curl_standard = f'''
# 标准TTS API - 多语言混合
curl -X POST "{BASE_URL}/tts" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "text": "你好world，这是一个test。",
    "text_lang": "auto",
    "ref_audio_path": "voice/vivienne/sample.mp3",
    "prompt_text": "Hello, this is a sample text.",
    "prompt_lang": "zh",
    "temperature": 1.0,
    "speed_factor": 1.0,
    "sample_steps": 32,
    "media_type": "wav"
  }}' \\
  --output "output.wav"
'''
    
    # 缓存音色API
    curl_cached = f'''
# 缓存音色API - 更快速度
curl -X POST "{BASE_URL}/tts_with_cached_voice" \\
  -d "voice_name=vivienne" \\
  -d "text=你好world，这是多语言test。" \\
  -d "text_lang=auto" \\
  -d "temperature=1.0" \\
  -d "speed_factor=1.0" \\
  --output "output_cached.wav"
'''
    
    print("1. 标准TTS API:")
    print(curl_standard)
    print("\n2. 缓存音色API:")
    print(curl_cached)

if __name__ == "__main__":
    # 运行示例
    main()
    
    # 打印curl示例
    print_curl_examples()
    
    print("\n🎉 示例完成!")
    print(f"🌐 服务器: {BASE_URL}")
    print("📁 生成的音频文件: output_*.wav")