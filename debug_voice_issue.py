#!/usr/bin/env python3
"""
声音错误诊断脚本
"""
import requests
import json
import time

BASE_URL = "http://219.144.21.182:9880"

def check_service_mode():
    """检查服务模式"""
    print("🔍 检查服务模式...")
    try:
        response = requests.get(f"{BASE_URL}/concurrent_stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 服务模式: {stats.get('mode', 'unknown')}")
            print(f"🔢 GPU数量: {stats.get('gpu_count', 0)}")
            
            if stats.get('mode') == 'concurrent':
                print("✅ 当前运行在并发模式")
                return True
            else:
                print("📱 当前运行在单线程模式")
                return False
        else:
            print(f"❌ 无法获取并发统计: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 检查服务模式失败: {e}")
        return None

def check_voices():
    """检查可用音色"""
    print("\n🎨 检查可用音色...")
    try:
        response = requests.get(f"{BASE_URL}/voices", timeout=10)
        if response.status_code == 200:
            voices_data = response.json()
            voices = voices_data.get('voices', {})
            print(f"📋 可用音色数量: {len(voices)}")
            
            for voice_name, info in voices.items():
                audio_files = info.get('audio_files', [])
                text_files = info.get('text_files', [])
                print(f"   🎤 {voice_name}: {len(audio_files)}个音频文件, {len(text_files)}个文本文件")
                
            return voices
        else:
            print(f"❌ 获取音色列表失败: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ 检查音色失败: {e}")
        return {}

def test_simple_tts():
    """测试简单TTS请求"""
    print("\n🧪 测试简单TTS请求...")
    
    payload = {
        "text": "你好，这是测试",
        "text_lang": "zh",
        "ref_audio_path": "voice/vivienne/sample.mp3",
        "prompt_text": "你好，这是测试",
        "prompt_lang": "zh"
    }
    
    try:
        print(f"📤 发送请求: {json.dumps(payload, ensure_ascii=False)}")
        response = requests.post(f"{BASE_URL}/tts", json=payload, timeout=60)
        
        if response.status_code == 200:
            file_size = len(response.content)
            print(f"✅ TTS请求成功，文件大小: {file_size/1024:.1f}KB")
            
            # 保存测试文件
            with open("debug_test.wav", "wb") as f:
                f.write(response.content)
            print("💾 已保存为 debug_test.wav")
            return True
        else:
            error_msg = response.text
            try:
                error_data = response.json()
                error_msg = error_data.get('message', str(error_data))
            except:
                pass
            print(f"❌ TTS请求失败: {response.status_code} - {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ TTS请求异常: {e}")
        return False

def test_cached_voice():
    """测试缓存音色"""
    print("\n🧪 测试缓存音色...")
    
    payload = {
        "voice_name": "vivienne",
        "text": "你好，这是缓存音色测试",
        "text_lang": "zh"
    }
    
    try:
        print(f"📤 发送缓存音色请求: {json.dumps(payload, ensure_ascii=False)}")
        response = requests.post(f"{BASE_URL}/tts_with_cached_voice", json=payload, timeout=60)
        
        if response.status_code == 200:
            file_size = len(response.content)
            print(f"✅ 缓存音色请求成功，文件大小: {file_size/1024:.1f}KB")
            
            # 保存测试文件
            with open("debug_cached_test.wav", "wb") as f:
                f.write(response.content)
            print("💾 已保存为 debug_cached_test.wav")
            return True
        else:
            error_msg = response.text
            try:
                error_data = response.json()
                error_msg = error_data.get('message', str(error_data))
            except:
                pass
            print(f"❌ 缓存音色请求失败: {response.status_code} - {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ 缓存音色请求异常: {e}")
        return False

def check_voice_cache():
    """检查音色缓存状态"""
    print("\n💾 检查音色缓存状态...")
    try:
        response = requests.get(f"{BASE_URL}/voice_cache_status", timeout=10)
        if response.status_code == 200:
            cache_data = response.json()
            cached_voices = cache_data.get('cached_voices', [])
            cache_size = cache_data.get('cache_size', 0)
            
            print(f"📦 缓存的音色数量: {cache_size}")
            print(f"📋 缓存的音色: {cached_voices}")
            
            cache_info = cache_data.get('cache_info', {})
            for voice_name, info in cache_info.items():
                print(f"   🎤 {voice_name}: 音频大小{info['audio_size']/1024:.1f}KB, "
                      f"提示文本: {info['prompt_text'][:30]}...")
            
            return cache_data
        else:
            print(f"❌ 获取缓存状态失败: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ 检查缓存状态失败: {e}")
        return {}

def main():
    """主诊断函数"""
    print("🩺 TTS声音错误诊断")
    print("=" * 50)
    
    # 1. 检查服务模式
    is_concurrent = check_service_mode()
    
    # 2. 检查音色
    voices = check_voices()
    
    # 3. 检查音色缓存
    cache_status = check_voice_cache()
    
    # 4. 测试简单TTS
    simple_success = test_simple_tts()
    
    # 5. 测试缓存音色
    cached_success = test_cached_voice()
    
    print("\n" + "=" * 50)
    print("📊 诊断结果总结:")
    print(f"   服务模式: {'并发' if is_concurrent else '单线程' if is_concurrent is not None else '未知'}")
    print(f"   可用音色: {len(voices)}")
    print(f"   缓存音色: {cache_status.get('cache_size', 0)}")
    print(f"   简单TTS: {'✅ 成功' if simple_success else '❌ 失败'}")
    print(f"   缓存音色TTS: {'✅ 成功' if cached_success else '❌ 失败'}")
    
    if not simple_success and not cached_success:
        print("\n🚨 两种方式都失败，可能的问题:")
        print("   1. 并发模式下音色设置问题 (已修复)")
        print("   2. 音频文件路径错误")
        print("   3. GPU工作器初始化问题")
        print("   4. 模型加载问题")
        
        print("\n💡 建议解决方案:")
        print("   1. 重新上传修复后的代码")
        print("   2. 重启TTS服务")
        print("   3. 检查voice目录结构")
        print("   4. 检查GPU状态")
    elif simple_success and not cached_success:
        print("\n⚠️ 缓存音色功能有问题")
    elif not simple_success and cached_success:
        print("\n⚠️ 简单TTS功能有问题")
    else:
        print("\n🎉 所有测试都成功！")

if __name__ == "__main__":
    main()