#!/usr/bin/env python3
"""
GPT-SoVITS Client SDK - 基本使用示例
"""

import os
from gpt_sovits_client import GPTSoVITSClient, LanguageType, TextSplitMethod

def basic_usage_example():
    """基本使用示例"""
    
    # 初始化客户端
    client = GPTSoVITSClient(base_url="http://localhost:9880")
    
    # 检查服务健康状态
    if not client.health_check():
        print("❌ GPT-SoVITS服务未运行")
        return
    
    print("✅ GPT-SoVITS服务运行正常")
    
    # 参考音频文件
    ref_audio = "sample.wav"
    if not os.path.exists(ref_audio):
        print(f"❌ 参考音频文件不存在: {ref_audio}")
        return
    
    # 示例1: 简单中文TTS
    print("\n📝 示例1: 简单中文TTS")
    text1 = "你好，这是一个中文语音合成测试。"
    response1 = client.synthesize_text(
        text=text1,
        ref_audio_path=ref_audio,
        text_lang=LanguageType.ALL_ZH
    )
    
    if response1.success:
        print(f"✅ 成功: {response1.audio_path}")
        print(f"📊 文件大小: {response1.file_size/1024:.1f}KB")
    else:
        print(f"❌ 失败: {response1.message}")
    
    # 示例2: 英文TTS
    print("\n📝 示例2: 英文TTS")
    text2 = "Hello, this is an English text-to-speech test."
    response2 = client.synthesize_text(
        text=text2,
        ref_audio_path=ref_audio,
        text_lang=LanguageType.EN
    )
    
    if response2.success:
        print(f"✅ 成功: {response2.audio_path}")
        print(f"📊 文件大小: {response2.file_size/1024:.1f}KB")
    else:
        print(f"❌ 失败: {response2.message}")
    
    # 示例3: 中英文混合TTS
    print("\n📝 示例3: 中英文混合TTS")
    text3 = "Hello 你好 world 世界"
    response3 = client.synthesize_text(
        text=text3,
        ref_audio_path=ref_audio,
        text_lang=LanguageType.AUTO  # 自动检测
    )
    
    if response3.success:
        print(f"✅ 成功: {response3.audio_path}")
        print(f"📊 文件大小: {response3.file_size/1024:.1f}KB")
    else:
        print(f"❌ 失败: {response3.message}")
    
    # 示例4: 中日英混合TTS
    print("\n📝 示例4: 中日英混合TTS")
    text4 = "Hello こんにちは 你好 world 世界"
    response4 = client.synthesize_text(
        text=text4,
        ref_audio_path=ref_audio,
        text_lang=LanguageType.AUTO
    )
    
    if response4.success:
        print(f"✅ 成功: {response4.audio_path}")
        print(f"📊 文件大小: {response4.file_size/1024:.1f}KB")
    else:
        print(f"❌ 失败: {response4.message}")

if __name__ == "__main__":
    basic_usage_example() 