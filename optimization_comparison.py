#!/usr/bin/env python3
"""
CUDA优化前后性能对比测试
"""
import requests
import time
import statistics

BASE_URL = "http://219.144.21.182:9880"

def toggle_optimization(enable=True):
    """切换优化状态"""
    try:
        response = requests.post(f"{BASE_URL}/toggle_optimization", params={"enable": enable}, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"🔧 {result.get('message', '')}")
            time.sleep(2)  # 等待设置生效
            return True
    except Exception as e:
        print(f"❌ 切换优化失败: {e}")
    return False

def test_single_request(text, test_name=""):
    """单次TTS请求测试"""
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
        response = requests.post(f"{BASE_URL}/tts", json=test_data, timeout=120)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        if response.status_code == 200:
            file_size = len(response.content)
            return {
                "success": True,
                "duration": duration,
                "file_size": file_size,
                "chars_per_second": len(text) / duration
            }
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"❌ 请求失败: {e}")
        
    return {"success": False, "duration": duration}

def run_comparison_test():
    """运行对比测试"""
    # 测试文本
    test_texts = [
        ("短文本", "这是短文本测试"),
        ("中文本", "这是一个中等长度的测试文本，用于验证CUDA优化效果"),
        ("长文本", "这是一个较长的测试文本，包含更多内容来全面测试系统性能。我们通过这个测试来验证CUDA Graph和智能分块处理的实际效果。")
    ]
    
    print("🔥 CUDA优化前后性能对比测试")
    print("=" * 70)
    
    results = {}
    
    for text_name, text in test_texts:
        print(f"\n📝 测试文本: {text_name} ({len(text)}字符)")
        print("-" * 50)
        
        results[text_name] = {"without_opt": [], "with_opt": []}
        
        # 测试无优化版本 (3次)
        print("🔄 测试无优化版本...")
        toggle_optimization(enable=False)
        
        for i in range(3):
            print(f"   第{i+1}次: ", end="", flush=True)
            result = test_single_request(text)
            if result["success"]:
                results[text_name]["without_opt"].append(result["duration"])
                print(f"{result['duration']:.2f}s ({result['chars_per_second']:.1f}字符/s)")
            else:
                print("失败")
        
        time.sleep(3)  # 服务器休息
        
        # 测试优化版本 (3次)
        print("🚀 测试CUDA优化版本...")
        toggle_optimization(enable=True)
        
        for i in range(3):
            print(f"   第{i+1}次: ", end="", flush=True)
            result = test_single_request(text)
            if result["success"]:
                results[text_name]["with_opt"].append(result["duration"])
                print(f"{result['duration']:.2f}s ({result['chars_per_second']:.1f}字符/s)")
            else:
                print("失败")
        
        time.sleep(2)  # 服务器休息
    
    # 统计分析
    print("\n📊 性能对比结果")
    print("=" * 70)
    
    for text_name, data in results.items():
        if data["without_opt"] and data["with_opt"]:
            avg_without = statistics.mean(data["without_opt"])
            avg_with = statistics.mean(data["with_opt"])
            speedup = avg_without / avg_with
            improvement = ((speedup - 1) * 100)
            
            print(f"\n{text_name}:")
            print(f"  无优化平均: {avg_without:.2f}s")
            print(f"  有优化平均: {avg_with:.2f}s") 
            print(f"  性能提升: {speedup:.2f}x ({improvement:.1f}%)")
            
            if improvement > 0:
                print(f"  ✅ 优化有效，提升 {improvement:.1f}%")
            else:
                print(f"  ⚠️ 优化效果不明显")

if __name__ == "__main__":
    run_comparison_test()