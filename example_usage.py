#!/usr/bin/env python3
"""
TTS客户端使用示例
"""
from tts_client import TTSClient

def main():
    # 创建客户端
    client = TTSClient("http://219.144.21.182:9880")
    
    # 检查API状态
    if not client.check_health():
        print("❌ API服务未运行，请先启动: python tts_api.py")
        return
    
    print("✅ API服务正常运行")
    
    # 获取voice列表
    voices_result = client.list_voices()
    voices = voices_result.get("voices", {})
    
    if not voices:
        print("❌ 没有找到可用的voice")
        return
    
    print(f"📋 可用voice: {list(voices.keys())}")
    
    # 使用voice1进行TTS
    voice_name = "voice1"
    text = "你好，这是一个测试，很高兴认识你！"
    
    print(f"\n🎵 使用 {voice_name} 生成语音...")
    
    # 方法1: 使用voice进行TTS（推荐）
    audio_data = client.tts_with_voice(
        voice_name=voice_name,
        text=text,
        output_file="output_voice1.wav"
    )
    
    if audio_data:
        print("✅ 语音生成成功!")
    else:
        print("❌ 语音生成失败!")

if __name__ == "__main__":
    main() 