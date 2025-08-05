#!/usr/bin/env python3
"""
最终优化测试 - 验证所有性能改进
"""
import requests
import time
import json

BASE_URL = "http://219.144.21.182:9880"

def test_optimization_status():
    """测试优化状态"""
    print("🔍 检查优化状态...")
    try:
        response = requests.get(f"{BASE_URL}/optimization_status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print("✅ 优化状态检查成功:")
            print(f"   CUDA可用: {status['cuda_available']}")
            print(f"   优化启用: {status['optimization_enabled']}")
            print(f"   优化初始化: {status['optimization_initialized']}")
            print(f"   配置: chunk_size={status['config']['chunk_size']}, overlap_ratio={status['config']['overlap_ratio']}")
            
            if 'optimization_details' in status:
                details = status['optimization_details']
                print(f"   缓存的Graph形状数: {details['cached_graph_shapes']}")
                print(f"   预热完成: {details['warmup_completed']}")
                print(f"   性能历史长度: {details['performance_history_length']}")
                print(f"   平均块处理时间: {details['avg_chunk_processing_time']:.3f}s")
            
            return status
        else:
            print(f"❌ 优化状态检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 优化状态检查异常: {e}")
    return None

def test_performance_comparison():
    """性能对比测试"""
    print("\n🚀 开始性能对比测试...")
    
    test_cases = [
        {"name": "超短文本", "text": "测试", "expected_improvement": "低"},
        {"name": "短文本", "text": "这是一个短文本测试", "expected_improvement": "中"},
        {"name": "中等文本", "text": "这是一个中等长度的测试文本，用于验证CUDA优化的实际效果和性能提升", "expected_improvement": "高"},
        {"name": "长文本", "text": "这是一个较长的测试文本，包含更多内容来全面测试系统性能。我们通过这个测试来验证CUDA Graph和智能分块处理的实际效果，同时观察自适应chunk_size调整和重叠处理的性能表现。", "expected_improvement": "高"}
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📝 测试: {test_case['name']} ({len(test_case['text'])}字符)")
        print(f"   预期改进: {test_case['expected_improvement']}")
        
        case_results = {"name": test_case["name"], "text_length": len(test_case["text"])}
        
        # 测试3次并取平均值
        durations = []
        for run in range(3):
            start_time = time.perf_counter()
            
            try:
                payload = {
                    "text": test_case["text"],
                    "text_lang": "zh",
                    "ref_audio_path": "voice/vivienne/sample.mp3",
                    "prompt_text": "Hello, this is a sample text.",
                    "prompt_lang": "en",
                    "temperature": 1.0,
                    "speed_factor": 1.0,
                    "sample_steps": 32
                }
                
                response = requests.post(f"{BASE_URL}/tts", json=payload, timeout=120)
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                if response.status_code == 200:
                    durations.append(duration)
                    file_size = len(response.content)
                    chars_per_second = len(test_case["text"]) / duration
                    print(f"   第{run+1}次: {duration:.3f}s ({chars_per_second:.1f}字符/s, {file_size/1024:.1f}KB)")
                else:
                    print(f"   第{run+1}次: 失败 - {response.status_code}")
            except Exception as e:
                end_time = time.perf_counter()
                duration = end_time - start_time
                print(f"   第{run+1}次: 异常 - {duration:.3f}s, {e}")
            
            time.sleep(1)  # 避免过于频繁的请求
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            std_deviation = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
            
            case_results.update({
                "avg_duration": avg_duration,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "std_deviation": std_deviation,
                "avg_chars_per_second": len(test_case["text"]) / avg_duration,
                "success_rate": len(durations) / 3 * 100
            })
            
            print(f"   📊 统计: 平均{avg_duration:.3f}s, 最快{min_duration:.3f}s, 标准差{std_deviation:.3f}s")
            print(f"   🎯 性能: {case_results['avg_chars_per_second']:.1f}字符/s, 成功率{case_results['success_rate']:.0f}%")
        
        results.append(case_results)
        time.sleep(2)  # 测试间隔
    
    return results

def test_optimization_features():
    """测试优化功能"""
    print("\n🔧 测试优化功能...")
    
    # 测试优化开关
    print("🔄 测试优化开关...")
    try:
        # 禁用优化
        response = requests.post(f"{BASE_URL}/toggle_optimization", params={"enable": False}, timeout=10)
        if response.status_code == 200:
            print("✅ 成功禁用优化")
        
        time.sleep(2)
        
        # 重新启用优化
        response = requests.post(f"{BASE_URL}/toggle_optimization", params={"enable": True}, timeout=10)
        if response.status_code == 200:
            print("✅ 成功启用优化")
    except Exception as e:
        print(f"❌ 优化开关测试失败: {e}")

def generate_performance_report(results):
    """生成性能报告"""
    print("\n📊 性能测试报告")
    print("=" * 60)
    
    if not results:
        print("❌ 无有效测试结果")
        return
    
    print(f"{'测试名称':<12} {'文本长度':<8} {'平均耗时':<10} {'处理速度':<12} {'成功率':<8}")
    print("-" * 60)
    
    total_chars = 0
    total_time = 0
    
    for result in results:
        if 'avg_duration' in result:
            name = result['name']
            text_len = result['text_length']
            avg_dur = result['avg_duration']
            chars_per_sec = result['avg_chars_per_second']
            success_rate = result['success_rate']
            
            print(f"{name:<12} {text_len:<8} {avg_dur:<10.3f} {chars_per_sec:<12.1f} {success_rate:<8.0f}%")
            
            total_chars += text_len
            total_time += avg_dur
    
    if total_time > 0:
        overall_speed = total_chars / total_time
        print("-" * 60)
        print(f"整体性能: {overall_speed:.1f} 字符/秒")
        
        # 性能评估
        if overall_speed > 15:
            print("🏆 性能评级: 优秀")
        elif overall_speed > 10:
            print("🥇 性能评级: 良好") 
        elif overall_speed > 5:
            print("🥈 性能评级: 一般")
        else:
            print("🥉 性能评级: 需要改进")

def main():
    """主测试函数"""
    print("🚀 最终CUDA优化测试")
    print("=" * 50)
    
    # 1. 检查优化状态
    optimization_status = test_optimization_status()
    
    # 2. 测试优化功能
    test_optimization_features()
    
    # 3. 性能对比测试
    results = test_performance_comparison()
    
    # 4. 生成报告
    generate_performance_report(results)
    
    # 5. 获取最终性能统计
    print("\n📈 最终性能统计")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/performance_stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"总请求数: {stats.get('total_requests', 0)}")
            print(f"平均TTS时间: {stats.get('avg_tts_time', 0):.3f}s")
            print(f"平均T2S时间: {stats.get('avg_t2s_time', 0):.3f}s")
            print(f"平均Vocoder时间: {stats.get('avg_vocoder_time', 0):.3f}s")
            print(f"吞吐量: {stats.get('requests_per_second', 0):.3f} 请求/s")
    except Exception as e:
        print(f"❌ 获取统计失败: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 测试完成!")

if __name__ == "__main__":
    main()