#!/usr/bin/env python3
"""
简单的TTS客户端工具
支持多语言混合推理，使用auto语言检测
"""
import requests
import argparse
import time
import os

BASE_URL = "http://219.144.21.182:9880"

def main():
    parser = argparse.ArgumentParser(description="TTS客户端 - 支持多语言混合推理")
    parser.add_argument("text", nargs="?", help="要合成的文本")
    parser.add_argument("-v", "--voice", default="vivienne", 
                       choices=["vivienne", "allen"], 
                       help="音色选择 (默认: vivienne)")
    parser.add_argument("-l", "--lang", default="auto", 
                       choices=["auto", "zh", "en", "ja", "ko"],
                       help="文本语言 (默认: auto - 自动检测)")
    parser.add_argument("-t", "--temperature", type=float, default=1.0,
                       help="温度参数 (默认: 1.0)")
    parser.add_argument("-s", "--speed", type=float, default=1.0,
                       help="语速控制 (默认: 1.0)")
    parser.add_argument("-o", "--output", default="output.wav",
                       help="输出文件名 (默认: output.wav)")
    parser.add_argument("--cached", action="store_true",
                       help="使用缓存音色API (更快)")
    parser.add_argument("--status", action="store_true",
                       help="检查服务器和优化状态")
    
    args = parser.parse_args()
    
    if args.status:
        check_status()
        return
    
    if not args.text:
        parser.error("需要提供要合成的文本，或使用 --status 检查服务器状态")
    
    print(f"🎙️ TTS 客户端")
    print(f"📝 文本: {args.text}")
    print(f"🎨 音色: {args.voice}")
    print(f"🌍 语言: {args.lang}")
    print(f"🌡️ 温度: {args.temperature}")
    print(f"⚡ 语速: {args.speed}")
    print(f"💾 输出: {args.output}")
    print("-" * 50)
    
    success = generate_speech(
        text=args.text,
        voice=args.voice,
        lang=args.lang,
        temperature=args.temperature,
        speed=args.speed,
        output=args.output,
        use_cached=args.cached
    )
    
    if success:
        print(f"🎉 语音生成完成: {args.output}")
        print(f"📁 文件大小: {os.path.getsize(args.output)/1024:.1f}KB")
    else:
        print("❌ 语音生成失败")

def generate_speech(text, voice, lang, temperature, speed, output, use_cached=False):
    """生成语音"""
    start_time = time.time()
    
    try:
        if use_cached:
            # 使用缓存音色API
            print("🚀 使用缓存音色API...")
            params = {
                "voice_name": voice,
                "text": text,
                "text_lang": lang,
                "temperature": temperature,
                "speed_factor": speed,
                "sample_steps": 32
            }
            
            response = requests.post(
                f"{BASE_URL}/tts_with_cached_voice",
                params=params,
                timeout=120
            )
        else:
            # 使用标准TTS API
            print("🎯 使用标准TTS API...")
            prompt_texts = {
                "vivienne": "Hello, this is a sample text.",
                "allen": "你好，这是一个示例文本。"
            }
            
            payload = {
                "text": text,
                "text_lang": lang,
                "ref_audio_path": f"voice/{voice}/sample.mp3",
                "prompt_text": prompt_texts.get(voice, "Hello, this is a sample text."),
                "prompt_lang": "zh" if voice == "allen" else "en",
                "temperature": temperature,
                "speed_factor": speed,
                "sample_steps": 32,
                "top_k": 5,
                "top_p": 1.0,
                "text_split_method": "cut5",
                "batch_size": 1,
                "media_type": "wav",
                "streaming_mode": False
            }
            
            response = requests.post(
                f"{BASE_URL}/tts",
                json=payload,
                timeout=120
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            # 保存音频文件
            with open(output, "wb") as f:
                f.write(response.content)
            
            file_size = len(response.content)
            processing_speed = len(text) / duration
            
            print(f"✅ 生成成功!")
            print(f"⏱️ 耗时: {duration:.2f}秒")
            print(f"🚀 处理速度: {processing_speed:.1f} 字符/秒")
            print(f"📦 文件大小: {file_size/1024:.1f}KB")
            
            return True
        else:
            try:
                error_detail = response.json()
                error_msg = error_detail.get('message', 'Unknown error')
            except:
                error_msg = response.text[:100]
            
            print(f"❌ API调用失败: {response.status_code}")
            print(f"🔍 错误详情: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def check_status():
    """检查服务器和优化状态"""
    print("🔍 检查服务器状态...")
    
    # 检查基本连接
    try:
        response = requests.get(f"{BASE_URL}/performance_stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("✅ 服务器连接正常")
            print(f"📊 总请求数: {stats.get('total_requests', 0)}")
            print(f"⏱️ 平均TTS时间: {stats.get('avg_tts_time', 0):.3f}s")
            print(f"🚀 吞吐量: {stats.get('requests_per_second', 0):.3f} 请求/s")
        else:
            print(f"⚠️ 服务器响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接服务器: {e}")
        return
    
    # 检查优化状态
    try:
        response = requests.get(f"{BASE_URL}/optimization_status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print("\n🔧 CUDA优化状态:")
            print(f"   CUDA可用: {'✅' if status.get('cuda_available') else '❌'}")
            print(f"   优化启用: {'✅' if status.get('optimization_enabled') else '❌'}")
            print(f"   优化初始化: {'✅' if status.get('optimization_initialized') else '❌'}")
            
            config = status.get('config', {})
            print(f"   配置: chunk_size={config.get('chunk_size')}, overlap_ratio={config.get('overlap_ratio')}")
            
            details = status.get('optimization_details', {})
            if details:
                print(f"   详情: 缓存{details.get('cached_graph_shapes', 0)}个形状")
                print(f"   预热: {'完成' if details.get('warmup_completed') else '未完成'}")
                print(f"   平均块处理时间: {details.get('avg_chunk_processing_time', 0):.3f}s")
        else:
            print("⚠️ 无法获取优化状态 (可能是旧版本API)")
    except Exception as e:
        print(f"⚠️ 优化状态检查失败: {e}")
    
    # 检查可用音色
    try:
        response = requests.get(f"{BASE_URL}/voices", timeout=10)
        if response.status_code == 200:
            voices = response.json().get("voices", {})
            print(f"\n🎨 可用音色 ({len(voices)}个):")
            for name, info in voices.items():
                description = info.get('description', info.get('gender', 'Unknown'))
                print(f"   - {name}: {description}")
        else:
            print("⚠️ 无法获取音色列表")
    except Exception as e:
        print(f"⚠️ 音色列表获取失败: {e}")

if __name__ == "__main__":
    main()