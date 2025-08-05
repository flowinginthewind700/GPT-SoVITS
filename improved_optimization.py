#!/usr/bin/env python3
"""
改进的CUDA优化建议
"""
import requests
import time

BASE_URL = "http://219.144.21.182:9880"

def warm_up_optimization():
    """预热CUDA优化，提升后续性能"""
    print("🔥 预热CUDA优化...")
    
    warm_up_texts = [
        "预热测试一",
        "预热测试二，稍微长一点的文本",
        "预热测试三，这是一个更长的文本用于预热CUDA Graph和GPU内存"
    ]
    
    for i, text in enumerate(warm_up_texts, 1):
        print(f"   预热 {i}/3: ", end="", flush=True)
        
        test_data = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": "voice/vivienne/sample.mp3", 
            "prompt_text": "Hello, this is a sample text.",
            "prompt_lang": "en",
            "temperature": 1.0,
            "speed_factor": 1.0,
            "sample_steps": 32
        }
        
        start_time = time.perf_counter()
        try:
            response = requests.post(f"{BASE_URL}/tts", json=test_data, timeout=60)
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                print(f"✅ {end_time - start_time:.2f}s")
            else:
                print("❌ 失败")
        except:
            print("❌ 异常")
        
        time.sleep(1)

def test_optimized_performance():
    """测试预热后的性能"""
    print("\n🚀 测试预热后性能...")
    
    test_texts = [
        "现在测试优化后的性能",
        "经过预热的CUDA Graph应该表现更好",
        "这个较长的文本用来测试分块处理的效果，验证系统在处理长文本时的稳定性和速度提升"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}: {text[:20]}... ({len(text)}字符)")
        
        durations = []
        for run in range(3):
            test_data = {
                "text": text,
                "text_lang": "zh",
                "ref_audio_path": "voice/vivienne/sample.mp3",
                "prompt_text": "Hello, this is a sample text.", 
                "prompt_lang": "en",
                "temperature": 1.0,
                "speed_factor": 1.0,
                "sample_steps": 32
            }
            
            start_time = time.perf_counter()
            try:
                response = requests.post(f"{BASE_URL}/tts", json=test_data, timeout=60)
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                if response.status_code == 200:
                    durations.append(duration)
                    chars_per_sec = len(text) / duration
                    print(f"   第{run+1}次: {duration:.2f}s ({chars_per_sec:.1f}字符/s)")
                else:
                    print(f"   第{run+1}次: 失败")
            except Exception as e:
                print(f"   第{run+1}次: 异常 {e}")
            
            time.sleep(0.5)
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            avg_chars_per_sec = len(text) / avg_duration
            print(f"   平均: {avg_duration:.2f}s ({avg_chars_per_sec:.1f}字符/s)")

def get_optimization_recommendations():
    """获取优化建议"""
    try:
        response = requests.get(f"{BASE_URL}/performance_stats")
        if response.status_code == 200:
            stats = response.json()
            
            print("\n💡 优化建议:")
            print("=" * 40)
            
            avg_tts_time = stats.get('avg_tts_time', 0)
            requests_per_sec = stats.get('requests_per_second', 0)
            
            if avg_tts_time > 3:
                print("⚠️ 平均TTS时间较长，建议:")
                print("   - 减小chunk_size到300-400")
                print("   - 增加GPU预热次数")
                print("   - 检查GPU内存使用情况")
            
            if requests_per_sec < 0.5:
                print("⚠️ 吞吐量较低，建议:")
                print("   - 启用更积极的CUDA Graph缓存")
                print("   - 考虑使用模型量化(FP16)")
                print("   - 优化内存管理")
            
            if stats.get('cuda_optimization_enabled', False):
                print("✅ CUDA优化已启用")
                if not stats.get('optimization_initialized', False):
                    print("⚠️ 优化器未初始化，建议预热系统")
                else:
                    print("✅ 优化器已初始化")
            else:
                print("❌ CUDA优化未启用")
                
    except Exception as e:
        print(f"❌ 无法获取统计信息: {e}")

if __name__ == "__main__":
    print("🔧 CUDA优化性能改进测试")
    print("=" * 50)
    
    # 1. 预热优化
    warm_up_optimization()
    
    # 2. 测试预热后性能
    test_optimized_performance()
    
    # 3. 获取优化建议
    get_optimization_recommendations()
    
    print("\n" + "=" * 50)
    print("🎯 测试完成! 建议:")
    print("1. 预热系统可以提升后续性能")
    print("2. 可以考虑调整chunk_size参数")
    print("3. 长文本处理效果通常更明显")