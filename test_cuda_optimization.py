#!/usr/bin/env python3
"""
测试CUDA优化效果
"""
import requests
import time

def test_optimization():
    base_url = "http://127.0.0.1:9880"
    
    test_texts = [
        "短文本测试",
        "这是一个中等长度的测试文本，用于验证优化效果。",
        "这是一个非常长的测试文本，包含了更多的内容，用于测试系统在处理长文本时的性能表现。我们希望通过CUDA Graph和分块处理来显著提升处理速度。"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. 测试文本: {text}")
        
        start_time = time.perf_counter()
        
        params = {
            "voice_name": "vivienne",
            "text": text,
            "text_lang": "zh",
            "temperature": 1.0,
            "speed_factor": 1.0
        }
        
        try:
            response = requests.post(f"{base_url}/tts_with_cached_voice", params=params, timeout=30)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if response.status_code == 200:
                file_size = len(response.content)
                print(f"✅ 成功 - 耗时: {duration:.3f}s, 文件大小: {file_size/1024:.1f}KB")
                
                # 保存音频文件
                output_file = f"test_output_{i}.wav"
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"📁 音频已保存: {output_file}")
                
            else:
                print(f"❌ 失败 - 耗时: {duration:.3f}s, 状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(f"❌ 异常 - 耗时: {duration:.3f}s, 错误: {e}")

def test_standard_tts():
    """测试标准TTS接口"""
    base_url = "http://127.0.0.1:9880"
    
    test_data = {
        "text": "这是一个测试文本，用于验证CUDA优化效果",
        "text_lang": "zh",
        "ref_audio_path": "voice/vivienne/sample.mp3",
        "prompt_text": "Hello, this is a sample text.",
        "prompt_lang": "en",
        "temperature": 1.0,
        "speed_factor": 1.0,
        "sample_steps": 32
    }
    
    print("\n🔧 测试标准TTS接口:")
    start_time = time.perf_counter()
    
    try:
        response = requests.post(f"{base_url}/tts", json=test_data, timeout=30)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        if response.status_code == 200:
            file_size = len(response.content)
            print(f"✅ 成功 - 耗时: {duration:.3f}s, 文件大小: {file_size/1024:.1f}KB")
            
            # 保存音频文件
            with open("standard_tts_output.wav", "wb") as f:
                f.write(response.content)
            print(f"📁 音频已保存: standard_tts_output.wav")
            
        else:
            print(f"❌ 失败 - 耗时: {duration:.3f}s, 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"❌ 异常 - 耗时: {duration:.3f}s, 错误: {e}")

if __name__ == "__main__":
    print("🚀 开始测试CUDA优化效果")
    print("=" * 50)
    
    # 测试缓存语音接口
    test_optimization()
    
    # 测试标准TTS接口  
    test_standard_tts()
    
    print("\n" + "=" * 50)
    print("测试完成!")