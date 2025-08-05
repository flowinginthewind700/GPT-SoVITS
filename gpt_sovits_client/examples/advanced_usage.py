#!/usr/bin/env python3
"""
GPT-SoVITS Client SDK - 高级使用示例
"""

import os
from gpt_sovits_client import (
    GPTSoVITSClient, 
    TTSRequest, 
    LanguageType, 
    TextSplitMethod
)

def advanced_usage_example():
    """高级使用示例"""
    
    # 初始化客户端
    client = GPTSoVITSClient(base_url="http://localhost:9880", timeout=300)
    
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
    
    # 示例1: 使用TTSRequest对象进行精确控制
    print("\n📝 示例1: 使用TTSRequest对象")
    request1 = TTSRequest(
        text="这是一个使用TTSRequest对象的测试，可以精确控制所有参数。",
        ref_audio_path=ref_audio,
        text_lang=LanguageType.ALL_ZH,
        text_split_method=TextSplitMethod.CUT5,
        top_k=5,
        top_p=1.0,
        temperature=1.0,
        speed_factor=1.0,
        repetition_penalty=1.35,
        parallel_infer=True
    )
    
    response1 = client.synthesize(request1)
    if response1.success:
        print(f"✅ 成功: {response1.audio_path}")
        print(f"📊 文件大小: {response1.file_size/1024:.1f}KB")
    else:
        print(f"❌ 失败: {response1.message}")
    
    # 示例2: 语言检测功能
    print("\n📝 示例2: 语言检测功能")
    test_texts = [
        "Hello world",
        "你好世界",
        "こんにちは世界",
        "안녕하세요 세계",
        "Hello 你好 world 世界",
        "Hello こんにちは 你好 안녕하세요"
    ]
    
    for text in test_texts:
        segments = client.detect_language_segments(text)
        is_mixed = client.is_mixed_language(text)
        primary_lang = client.get_primary_language(text)
        auto_lang = client.auto_detect_language(text)
        
        print(f"文本: {text}")
        print(f"  语言片段: {segments}")
        print(f"  混合语言: {is_mixed}")
        print(f"  主要语言: {primary_lang}")
        print(f"  自动检测: {auto_lang.value}")
        print()
    
    # 示例3: 批量处理
    print("\n📝 示例3: 批量处理")
    requests = [
        TTSRequest(
            text="第一个测试文本",
            ref_audio_path=ref_audio,
            text_lang=LanguageType.ALL_ZH
        ),
        TTSRequest(
            text="Second test text",
            ref_audio_path=ref_audio,
            text_lang=LanguageType.EN
        ),
        TTSRequest(
            text="Hello 你好 world 世界",
            ref_audio_path=ref_audio,
            text_lang=LanguageType.AUTO
        )
    ]
    
    responses = client.batch_synthesize(requests)
    for i, response in enumerate(responses, 1):
        if response.success:
            print(f"✅ 批量任务{i}成功: {response.audio_path}")
        else:
            print(f"❌ 批量任务{i}失败: {response.message}")
    
    # 示例4: 错误处理
    print("\n📝 示例4: 错误处理")
    
    # 测试不存在的文件
    response_error1 = client.synthesize_text(
        text="测试错误处理",
        ref_audio_path="nonexistent.wav",
        text_lang=LanguageType.ALL_ZH
    )
    print(f"文件不存在错误: {response_error1.message}")
    
    # 测试空文本
    response_error2 = client.synthesize_text(
        text="",
        ref_audio_path=ref_audio,
        text_lang=LanguageType.ALL_ZH
    )
    print(f"空文本错误: {response_error2.message}")

if __name__ == "__main__":
    advanced_usage_example() 