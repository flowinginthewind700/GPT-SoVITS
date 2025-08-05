#!/usr/bin/env python3
"""
详细的性能分析脚本
"""
import requests
import json
import time
import numpy as np

def analyze_performance():
    base_url = "http://219.144.21.182:9880"
    
    print("🔍 详细性能分析开始...")
    
    # 1. 重置性能统计
    print("\n1. 重置性能统计...")
    try:
        response = requests.post(f"{base_url}/reset_performance_stats")
        if response.status_code == 200:
            print("   ✅ 性能统计已重置")
    except Exception as e:
        print(f"❌ 重置性能统计失败: {e}")
    
    # 2. 刷新缓存
    print("\n2. 刷新音色缓存...")
    try:
        response = requests.post(f"{base_url}/refresh_voice_cache")
        if response.status_code == 200:
            result = response.json()
            cache_result = result.get('result', {})
            print(f"   ✅ 缓存刷新成功: {cache_result.get('success_count', 0)} 个音色")
    except Exception as e:
        print(f"❌ 缓存刷新异常: {e}")
    
    # 3. 进行多次测试
    test_cases = [
        {"text": "你好", "desc": "短文本"},
        {"text": "你好，这是性能测试。", "desc": "中等文本"},
        {"text": "这是一个较长的测试文本，用于测试不同长度文本的性能表现。", "desc": "长文本"},
        {"text": "这是一个非常长的测试文本，包含了更多的内容，用于测试系统在处理长文本时的性能表现。", "desc": "超长文本"}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试: {test_case['desc']}")
        print(f"   文本: {test_case['text']}")
        
        try:
            start_time = time.perf_counter()
            
            params = {
                "voice_name": "vivienne",
                "text": test_case["text"],
                "text_lang": "zh",
                "temperature": 1.0,
                "speed_factor": 1.0
            }
            
            response = requests.post(f"{base_url}/tts_with_cached_voice", params=params, timeout=180)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if response.status_code == 200:
                file_size = len(response.content)
                print(f"   ✅ 成功 - 耗时: {duration:.3f}s, 文件大小: {file_size/1024:.1f}KB")
                
                # 保存音频文件
                with open(f"perf_analysis_{i}.wav", "wb") as f:
                    f.write(response.content)
                
                results.append({
                    "case": test_case["desc"],
                    "text_length": len(test_case["text"]),
                    "duration": duration,
                    "file_size": file_size,
                    "success": True
                })
            else:
                print(f"   ❌ 失败 - 耗时: {duration:.3f}s, 错误: {response.text}")
                results.append({
                    "case": test_case["desc"],
                    "text_length": len(test_case["text"]),
                    "duration": duration,
                    "file_size": 0,
                    "success": False
                })
                
        except Exception as e:
            print(f"   ❌ 异常 - {e}")
            results.append({
                "case": test_case["desc"],
                "text_length": len(test_case["text"]),
                "duration": 0,
                "file_size": 0,
                "success": False
            })
    
    # 4. 获取性能统计
    print("\n4. 获取性能统计...")
    try:
        response = requests.get(f"{base_url}/performance_stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   总请求数: {stats.get('total_requests', 0)}")
            print(f"   平均时间:")
            avg_times = stats.get('average_times', {})
            print(f"     - 总TTS: {avg_times.get('total_tts', 0)}s")
            print(f"     - T2S推理: {avg_times.get('t2s_inference', 0)}s")
            print(f"     - Vocoder合成: {avg_times.get('vocoder_synthesis', 0)}s")
            print(f"     - 音频后处理: {avg_times.get('audio_postprocess', 0)}s")
            
            # 性能分析
            total_tts = avg_times.get('total_tts', 0)
            t2s_time = avg_times.get('t2s_inference', 0)
            vocoder_time = avg_times.get('vocoder_synthesis', 0)
            postprocess_time = avg_times.get('audio_postprocess', 0)
            
            if total_tts > 0:
                print(f"\n📊 性能分析:")
                print(f"   T2S推理占比: {t2s_time/total_tts*100:.1f}%")
                print(f"   Vocoder合成占比: {vocoder_time/total_tts*100:.1f}%")
                print(f"   音频后处理占比: {postprocess_time/total_tts*100:.1f}%")
                
                # 性能建议
                print(f"\n💡 性能优化建议:")
                if postprocess_time > total_tts * 0.3:
                    print(f"   - 音频后处理耗时较长，建议优化音频拼接算法")
                if vocoder_time > total_tts * 0.5:
                    print(f"   - Vocoder合成耗时较长，建议使用更快的vocoder模型")
                if t2s_time > total_tts * 0.2:
                    print(f"   - T2S推理耗时较长，建议优化模型推理")
                
                # 吞吐量分析
                successful_results = [r for r in results if r['success']]
                if successful_results:
                    avg_duration = sum(r['duration'] for r in successful_results) / len(successful_results)
                    throughput = 1 / avg_duration if avg_duration > 0 else 0
                    print(f"\n🚀 吞吐量分析:")
                    print(f"   平均处理时间: {avg_duration:.3f}s")
                    print(f"   理论最大吞吐量: {throughput:.2f} 请求/秒")
                    
                    # 文本长度与处理时间的关系
                    text_lengths = [r['text_length'] for r in successful_results]
                    durations = [r['duration'] for r in successful_results]
                    
                    if len(text_lengths) > 1:
                        correlation = np.corrcoef(text_lengths, durations)[0, 1]
                        print(f"   文本长度与处理时间相关性: {correlation:.3f}")
                        
                        if correlation > 0.7:
                            print(f"   - 文本长度与处理时间高度相关，建议优化长文本处理")
                        elif correlation < 0.3:
                            print(f"   - 文本长度与处理时间相关性较低，系统性能稳定")
                        else:
                            print(f"   - 文本长度与处理时间中等相关")
    except Exception as e:
        print(f"❌ 获取性能统计失败: {e}")
    
    # 5. 生成性能报告
    print("\n5. 生成性能报告...")
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n📈 测试结果汇总:")
        print(f"   成功测试数: {len(successful_results)}/{len(results)}")
        print(f"   平均处理时间: {sum(r['duration'] for r in successful_results)/len(successful_results):.3f}s")
        print(f"   最短处理时间: {min(r['duration'] for r in successful_results):.3f}s")
        print(f"   最长处理时间: {max(r['duration'] for r in successful_results):.3f}s")
        print(f"   平均文件大小: {sum(r['file_size'] for r in successful_results)/len(successful_results)/1024:.1f}KB")
        
        # 详细结果
        print(f"\n📋 详细结果:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['case']}: {result['duration']:.3f}s ({result['file_size']/1024:.1f}KB)")
    
    print("\n🎉 性能分析完成!")

if __name__ == "__main__":
    analyze_performance() 