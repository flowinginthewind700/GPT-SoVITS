#!/usr/bin/env python3
"""
快速修复：禁用并发模式，恢复单线程稳定服务
"""
import sys
import os

def disable_concurrent_mode():
    """禁用并发模式"""
    
    # 1. 修改配置文件
    config_file = "GPT_SoVITS/configs/tts_infer.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 禁用并发
        content = content.replace("enable_concurrent_tts: true", "enable_concurrent_tts: false")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已禁用配置文件中的并发模式: {config_file}")
    else:
        print(f"⚠️ 配置文件不存在: {config_file}")
    
    # 2. 修改tts_api.py，强制禁用并发
    api_file = "tts_api.py"
    if os.path.exists(api_file):
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并发检查的代码段
        old_concurrent_check = """if CONCURRENT_TTS_AVAILABLE and enable_concurrent and torch.cuda.device_count() >= 2:
    print(f"🚀 启用并发TTS模式，使用 {num_gpus} 个GPU")
    CONCURRENT_MODE = True"""
        
        new_concurrent_check = """if False:  # 临时禁用并发模式
    print(f"🚀 启用并发TTS模式，使用 {num_gpus} 个GPU")
    CONCURRENT_MODE = True"""
        
        if old_concurrent_check in content:
            content = content.replace(old_concurrent_check, new_concurrent_check)
            
            with open(api_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 已强制禁用API文件中的并发模式: {api_file}")
        else:
            print("⚠️ 未找到并发检查代码，可能已经修改过")
    else:
        print(f"⚠️ API文件不存在: {api_file}")

def main():
    print("🔧 快速修复：禁用并发模式")
    print("=" * 40)
    
    disable_concurrent_mode()
    
    print("\n✅ 修复完成！")
    print("\n🚀 下一步操作:")
    print("1. 重启TTS服务: python tts_api.py -a 0.0.0.0 -p 9880")
    print("2. 测试服务: python debug_voice_issue.py")
    print("3. 或者直接测试: python simple_tts_client.py '测试文本' -v vivienne")
    
    print("\n📝 说明:")
    print("- 现在服务将运行在稳定的单线程模式")
    print("- 虽然没有并发处理，但声音应该恢复正常")
    print("- 稍后可以再调试并发模式的问题")

if __name__ == "__main__":
    main()