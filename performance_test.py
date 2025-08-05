#!/usr/bin/env python3
"""
性能对比测试 - CUDA优化 vs 原始版本
"""
import requests
import time
import json

BASE_URL = "http://127.0.0.1:9880"

def get_performance_stats():
    """获取性能统计"""
    try:
        response = requests.get(f"{BASE_URL}/performance_stats")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def toggle_optimization(enable=True):
    """切换优化状态"""
    try:
        response = requests.post(f"{BASE_URL}/toggle_optimization", params={"enable": enable})
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def test_tts_performance(text, optimization_enabled=True):
    """测试TTS性能"""
    # 切换优化状态
    toggle_result = toggle_optimization(optimization_enabled)
    if toggle_result:
        print(f"📊 {toggle_result['message']}")
    
    # 等待一秒让设置生效
    time.sleep(1)
    
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
            file_size = len(response.content)
            return {
                "success": True,
                "duration": duration,
                "file_size": file_size,
                "optimization_enabled": optimization_enabled
            }
        else:
            return {
                "success": False,
                "duration": duration,
                "error": response.text,
                "optimization_enabled": optimization_enabled
            }
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        return {
            "success": False,
            "duration": duration,
            "error": str(e),
            "optimization_enabled": optimization_enabled
        }

def run_comparison_test():
    """运行对比测试"""
    test_texts = [
        "短文本测试",
        "这是一个中等长度的测试文本，用于验证优化效果。测试系统在处理中等长度文本时的性能。",
        "这是一个非常长的测试文本，包含了更多的内容，用于测试系统在处理长文本时的性能表现。我们希望通过CUDA Graph和分块处理来显著提升处理速度。这个测试将帮助我们验证优化的实际效果，并确保系统能够稳定高效地处理各种长度的文本输入。"
    ]
    
    print("🚀 开始性能对比测试")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}: {text[:30]}{'...' if len(text) > 30 else ''}")
        print("-" * 40)
        
        # 测试未优化版本
        print("🔄 测试原始版本...")
        original_result = test_tts_performance(text, optimization_enabled=False)
        
        # 测试优化版本
        print("🔄 测试CUDA优化版本...")
        optimized_result = test_tts_performance(text, optimization_enabled=True)
        
        # 显示结果
        if original_result["success"] and optimized_result["success"]:
            speedup = original_result["duration"] / optimized_result["duration"]
            print(f"📊 结果对比:")
            print(f"  原始版本: {original_result['duration']:.3f}s")
            print(f"  优化版本: {optimized_result['duration']:.3f}s")
            print(f"  性能提升: {speedup:.2f}x ({((speedup-1)*100):.1f}%)")
            print(f"  文件大小: {optimized_result['file_size']/1024:.1f}KB")
        else:
            print("❌ 测试失败:")
            if not original_result["success"]:
                print(f"  原始版本错误: {original_result['error']}")
            if not optimized_result["success"]:
                print(f"  优化版本错误: {optimized_result['error']}")
    
    # 显示总体统计
    print("\n" + "=" * 60)
    print("📈 总体性能统计:")
    stats = get_performance_stats()
    if stats:
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  平均TTS时间: {stats['avg_tts_time']:.3f}s")
        print(f"  平均T2S时间: {stats['avg_t2s_time']:.3f}s")
        print(f"  平均Vocoder时间: {stats['avg_vocoder_time']:.3f}s")
        print(f"  平均吞吐量: {stats['requests_per_second']:.3f} 请求/秒")
        print(f"  CUDA优化状态: {'启用' if stats['cuda_optimization_enabled'] else '禁用'}")
        print(f"  优化器初始化: {'是' if stats['optimization_initialized'] else '否'}")

if __name__ == "__main__":
    run_comparison_test()