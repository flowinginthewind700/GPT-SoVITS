#!/usr/bin/env python3
"""
并发TTS测试脚本 - 验证多GPU并发处理能力
"""
import asyncio
import aiohttp
import time
import json
from concurrent.futures import ThreadPoolExecutor
import threading

BASE_URL = "http://219.144.21.182:9880"

async def single_tts_request(session, text, voice="vivienne", request_id=None):
    """单个TTS请求"""
    payload = {
        "text": text,
        "text_lang": "auto",
        "ref_audio_path": f"voice/{voice}/sample.mp3",
        "prompt_text": "Hello, this is a sample text." if voice == "vivienne" else "你好，这是一个示例文本。",
        "prompt_lang": "en" if voice == "vivienne" else "zh",
        "temperature": 1.0,
        "speed_factor": 1.0,
        "sample_steps": 32
    }
    
    start_time = time.perf_counter()
    
    try:
        async with session.post(f"{BASE_URL}/tts", json=payload, timeout=120) as response:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if response.status == 200:
                content = await response.read()
                return {
                    "success": True,
                    "request_id": request_id,
                    "duration": duration,
                    "file_size": len(content),
                    "text_length": len(text),
                    "chars_per_second": len(text) / duration,
                    "voice": voice
                }
            else:
                error_text = await response.text()
                return {
                    "success": False,
                    "request_id": request_id,
                    "duration": duration,
                    "error": f"HTTP {response.status}: {error_text[:100]}",
                    "voice": voice
                }
                
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        return {
            "success": False,
            "request_id": request_id,
            "duration": duration,
            "error": str(e),
            "voice": voice
        }

async def check_concurrent_status():
    """检查并发状态"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/concurrent_stats", timeout=10) as response:
                if response.status == 200:
                    stats = await response.json()
                    print(f"📊 服务模式: {stats.get('mode', 'unknown')}")
                    print(f"🔢 GPU数量: {stats.get('gpu_count', 0)}")
                    
                    if stats.get('mode') == 'concurrent':
                        print(f"📈 总请求数: {stats.get('total_requests', 0)}")
                        print(f"✅ 完成请求数: {stats.get('completed_requests', 0)}")
                        print(f"❌ 失败请求数: {stats.get('failed_requests', 0)}")
                        print(f"📊 成功率: {stats.get('success_rate', 0):.1f}%")
                        
                        # GPU统计
                        gpu_stats = stats.get('gpu_stats', {})
                        for gpu_id, gpu_stat in gpu_stats.items():
                            print(f"   GPU {gpu_id}: {gpu_stat['requests_processed']}个请求, "
                                  f"平均{gpu_stat['average_processing_time']:.3f}s")
                        
                        # 队列状态
                        queue_status = stats.get('queue_status', {})
                        for gpu_name, queue_info in queue_status.items():
                            print(f"   {gpu_name}: 请求队列{queue_info['request_queue']}, "
                                  f"响应队列{queue_info['response_queue']}")
                    
                    return stats
                else:
                    print(f"❌ 获取并发统计失败: {response.status}")
                    return None
    except Exception as e:
        print(f"❌ 检查并发状态失败: {e}")
        return None

async def concurrent_load_test(num_requests=20, max_concurrent=10):
    """并发负载测试"""
    print(f"🚀 开始并发负载测试")
    print(f"📊 总请求数: {num_requests}, 最大并发数: {max_concurrent}")
    
    # 准备测试文本
    test_texts = [
        "你好world，这是并发测试1！",
        "Hello世界，concurrent test 2！",
        "这是一个较长的并发测试文本3，包含更多内容来验证系统性能。",
        "Concurrent processing test 4 with mixed languages！",
        "并发处理测试5，验证多GPU负载均衡效果。",
        "Multi-GPU concurrent test 6 for performance validation！",
        "这是第7个并发测试，检查系统稳定性和吞吐量。",
        "Parallel processing test 8 with auto language detection！"
    ]
    
    voices = ["vivienne", "allen"]
    
    # 创建请求列表
    requests = []
    for i in range(num_requests):
        text = test_texts[i % len(test_texts)]
        voice = voices[i % len(voices)]
        requests.append((text, voice, f"req_{i+1}"))
    
    # 执行并发测试
    start_time = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(text, voice, req_id):
            async with semaphore:
                return await single_tts_request(session, text, voice, req_id)
        
        # 执行所有请求
        tasks = [limited_request(text, voice, req_id) for text, voice, req_id in requests]
        results = await asyncio.gather(*tasks)
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    
    # 分析结果
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n📊 并发测试结果:")
    print(f"   总耗时: {total_duration:.3f}秒")
    print(f"   成功请求: {len(successful_results)}/{num_requests}")
    print(f"   失败请求: {len(failed_results)}/{num_requests}")
    print(f"   成功率: {len(successful_results)/num_requests*100:.1f}%")
    print(f"   平均吞吐量: {num_requests/total_duration:.3f} 请求/秒")
    
    if successful_results:
        avg_duration = sum(r['duration'] for r in successful_results) / len(successful_results)
        avg_chars_per_sec = sum(r['chars_per_second'] for r in successful_results) / len(successful_results)
        min_duration = min(r['duration'] for r in successful_results)
        max_duration = max(r['duration'] for r in successful_results)
        
        print(f"   平均请求时间: {avg_duration:.3f}秒")
        print(f"   最快请求: {min_duration:.3f}秒")
        print(f"   最慢请求: {max_duration:.3f}秒")
        print(f"   平均处理速度: {avg_chars_per_sec:.1f} 字符/秒")
    
    if failed_results:
        print(f"\n❌ 失败请求详情:")
        for result in failed_results[:5]:  # 只显示前5个错误
            print(f"   {result['request_id']}: {result['error']}")
    
    return results

async def sequential_test(num_requests=5):
    """顺序测试作为对比"""
    print(f"\n🔄 开始顺序测试 (对比基准)")
    
    test_texts = [
        "顺序测试1，验证单请求性能",
        "Sequential test 2 for baseline",
        "第3个顺序测试，用于性能对比",
        "Fourth sequential test for comparison",
        "最后一个顺序测试，建立基准线"
    ]
    
    start_time = time.perf_counter()
    results = []
    
    async with aiohttp.ClientSession() as session:
        for i, text in enumerate(test_texts[:num_requests]):
            voice = "vivienne" if i % 2 == 0 else "allen"
            result = await single_tts_request(session, text, voice, f"seq_{i+1}")
            results.append(result)
            print(f"   完成 {i+1}/{num_requests}: {result['duration']:.3f}s")
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    
    successful_results = [r for r in results if r['success']]
    
    print(f"\n📊 顺序测试结果:")
    print(f"   总耗时: {total_duration:.3f}秒")
    print(f"   成功率: {len(successful_results)}/{num_requests}")
    print(f"   平均吞吐量: {num_requests/total_duration:.3f} 请求/秒")
    
    if successful_results:
        avg_duration = sum(r['duration'] for r in successful_results) / len(successful_results)
        print(f"   平均请求时间: {avg_duration:.3f}秒")
    
    return results, total_duration

async def main():
    """主测试函数"""
    print("🔥 TTS并发性能测试")
    print("=" * 60)
    
    # 1. 检查服务状态
    print("1. 检查服务状态")
    status = await check_concurrent_status()
    if not status:
        print("❌ 服务器无法访问，测试终止")
        return
    
    print("\n" + "=" * 60)
    
    # 2. 顺序测试 (建立基准)
    print("2. 顺序测试 (建立基准)")
    sequential_results, sequential_time = await sequential_test(5)
    
    print("\n" + "=" * 60)
    
    # 3. 并发测试
    print("3. 并发负载测试")
    concurrent_results = await concurrent_load_test(20, 10)
    
    print("\n" + "=" * 60)
    
    # 4. 最终统计
    print("4. 最终统计对比")
    final_status = await check_concurrent_status()
    
    # 性能对比
    if sequential_results and concurrent_results:
        sequential_success = [r for r in sequential_results if r['success']]
        concurrent_success = [r for r in concurrent_results if r['success']]
        
        if sequential_success and concurrent_success:
            seq_avg_time = sum(r['duration'] for r in sequential_success) / len(sequential_success)
            conc_avg_time = sum(r['duration'] for r in concurrent_success) / len(concurrent_success)
            
            seq_throughput = len(sequential_success) / sequential_time
            conc_throughput = len(concurrent_success) / sum(r['duration'] for r in concurrent_success) * len(concurrent_success)
            
            print(f"\n🏆 性能对比:")
            print(f"   顺序处理平均时间: {seq_avg_time:.3f}s")
            print(f"   并发处理平均时间: {conc_avg_time:.3f}s")
            print(f"   时间效率提升: {seq_avg_time/conc_avg_time:.2f}x")
            print(f"   吞吐量提升: {conc_throughput/seq_throughput:.2f}x")
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    asyncio.run(main())