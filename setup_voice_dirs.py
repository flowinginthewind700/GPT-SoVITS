#!/usr/bin/env python3
"""
设置voice目录的辅助脚本
"""
import os
import shutil
import json

def setup_voice_directories():
    """设置voice目录结构"""
    
    # 检查是否有现有的音频文件可以复制
    possible_audio_files = [
        "kokoro_tts_test_20250803_110326.wav",
        "sample.wav",
        "test1_output.wav",
        "simple_tts_output.wav"
    ]
    
    available_audio = None
    for audio_file in possible_audio_files:
        if os.path.exists(audio_file):
            available_audio = audio_file
            break
    
    if not available_audio:
        print("❌ 没有找到可用的音频文件")
        print("请将音频文件复制到voice目录中，或者运行此脚本前先准备音频文件")
        return
    
    print(f"✅ 找到音频文件: {available_audio}")
    
    # 设置voice目录
    voice_dirs = ["voice1", "voice2", "voice3"]
    
    for voice_name in voice_dirs:
        voice_path = os.path.join("voice", voice_name)
        
        # 确保目录存在
        os.makedirs(voice_path, exist_ok=True)
        
        # 复制音频文件
        audio_file = os.path.join(voice_path, "sample.wav")
        if not os.path.exists(audio_file):
            shutil.copy2(available_audio, audio_file)
            print(f"✅ 复制音频文件到 {voice_path}/sample.wav")
        
        # 确保配置文件存在
        config_file = os.path.join(voice_path, "config.json")
        if not os.path.exists(config_file):
            if voice_name == "voice1":
                config = {
                    "name": "voice1",
                    "gender": "female",
                    "description": "温柔女声",
                    "language": "zh"
                }
            elif voice_name == "voice2":
                config = {
                    "name": "voice2",
                    "gender": "male",
                    "description": "磁性男声",
                    "language": "zh"
                }
            else:
                config = {
                    "name": voice_name,
                    "gender": "unknown",
                    "description": f"{voice_name}音色",
                    "language": "zh"
                }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✅ 创建配置文件 {config_file}")
        
        # 确保文本文件存在
        text_file = os.path.join(voice_path, "sample.wav.txt")
        if not os.path.exists(text_file):
            if voice_name == "voice1":
                text_content = "你好，我是温柔女声，很高兴认识你。"
            elif voice_name == "voice2":
                text_content = "你好，我是磁性男声，很高兴认识你。"
            else:
                text_content = f"你好，我是{voice_name}，很高兴认识你。"
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"✅ 创建文本文件 {text_file}")
    
    print("\n🎉 Voice目录设置完成!")
    print("现在你可以运行: python tts_client.py")

if __name__ == "__main__":
    setup_voice_directories() 