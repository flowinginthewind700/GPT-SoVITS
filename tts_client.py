#!/usr/bin/env python3
"""
TTS客户端示例
支持voice管理和简单的TTS推理
"""
import requests
import json
import os
import time
from typing import Optional, Dict, Any


class TTSClient:
    def __init__(self, base_url: str = "http://219.144.21.182:9880"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        """检查API健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/voices", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_voices(self) -> Dict[str, Any]:
        """获取所有可用的voice列表"""
        try:
            response = self.session.get(f"{self.base_url}/voices")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ 获取voice列表失败: {response.text}")
                return {"voices": {}}
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return {"voices": {}}
    
    def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """获取指定voice的详细信息"""
        try:
            response = self.session.get(f"{self.base_url}/voice/{voice_name}")
            if response.status_code == 200:
                return response.json()["voice"]
            else:
                print(f"❌ 获取voice信息失败: {response.text}")
                return None
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return None
    
    def set_voice(self, voice_name: str, audio_file: str = None) -> bool:
        """设置指定的voice为当前参考音频"""
        try:
            params = {}
            if audio_file:
                params["audio_file"] = audio_file
            
            response = self.session.post(f"{self.base_url}/voice/{voice_name}/set", params=params)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 设置voice成功: {voice_name}")
                print(f"   音频文件: {result['audio_path']}")
                print(f"   提示文本: {result['prompt_text']}")
                print(f"   性别: {result['gender']}")
                return True
            else:
                print(f"❌ 设置voice失败: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return False
    
    def tts_with_voice(self, voice_name: str, text: str, output_file: str = None, **kwargs) -> Optional[bytes]:
        """使用指定voice进行TTS推理"""
        try:
            # 构建请求参数
            params = {
                "voice_name": voice_name,
                "text": text,
                "text_lang": kwargs.get("text_lang", "zh"),
                "audio_file": kwargs.get("audio_file"),
                "top_k": kwargs.get("top_k", 5),
                "top_p": kwargs.get("top_p", 1.0),
                "temperature": kwargs.get("temperature", 1.0),
                "text_split_method": kwargs.get("text_split_method", "cut5"),
                "batch_size": kwargs.get("batch_size", 1),
                "batch_threshold": kwargs.get("batch_threshold", 0.75),
                "split_bucket": kwargs.get("split_bucket", False),
                "speed_factor": kwargs.get("speed_factor", 1.0),
                "fragment_interval": kwargs.get("fragment_interval", 0.3),
                "seed": kwargs.get("seed", -1),
                "media_type": kwargs.get("media_type", "wav"),
                "streaming_mode": kwargs.get("streaming_mode", False),
                "parallel_infer": kwargs.get("parallel_infer", True),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.35),
                "sample_steps": kwargs.get("sample_steps", 32),
                "super_sampling": kwargs.get("super_sampling", False)
            }
            
            print(f"🎵 开始TTS推理...")
            print(f"   Voice: {voice_name}")
            print(f"   文本: {text}")
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/tts_with_voice", params=params, timeout=180)
            end_time = time.time()
            
            if response.status_code == 200:
                audio_data = response.content
                duration = end_time - start_time
                file_size = len(audio_data)
                
                print(f"✅ TTS推理成功!")
                print(f"   耗时: {duration:.2f}秒")
                print(f"   文件大小: {file_size/1024:.1f}KB")
                
                # 保存音频文件
                if output_file:
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    print(f"   保存到: {output_file}")
                
                return audio_data
            else:
                print(f"❌ TTS推理失败: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return None
    
    def tts(self, text: str, ref_audio_path: str, prompt_text: str = "", output_file: str = None, **kwargs) -> Optional[bytes]:
        """传统的TTS推理（需要指定参考音频路径）"""
        try:
            # 构建请求参数
            req_data = {
                "text": text,
                "text_lang": kwargs.get("text_lang", "zh"),
                "ref_audio_path": ref_audio_path,
                "prompt_text": prompt_text,
                "prompt_lang": kwargs.get("prompt_lang", "zh"),
                "top_k": kwargs.get("top_k", 5),
                "top_p": kwargs.get("top_p", 1.0),
                "temperature": kwargs.get("temperature", 1.0),
                "text_split_method": kwargs.get("text_split_method", "cut5"),
                "batch_size": kwargs.get("batch_size", 1),
                "batch_threshold": kwargs.get("batch_threshold", 0.75),
                "split_bucket": kwargs.get("split_bucket", False),
                "speed_factor": kwargs.get("speed_factor", 1.0),
                "fragment_interval": kwargs.get("fragment_interval", 0.3),
                "seed": kwargs.get("seed", -1),
                "media_type": kwargs.get("media_type", "wav"),
                "streaming_mode": kwargs.get("streaming_mode", False),
                "parallel_infer": kwargs.get("parallel_infer", True),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.35),
                "sample_steps": kwargs.get("sample_steps", 32),
                "super_sampling": kwargs.get("super_sampling", False)
            }
            
            print(f"🎵 开始TTS推理...")
            print(f"   参考音频: {ref_audio_path}")
            print(f"   文本: {text}")
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/tts", json=req_data, timeout=180)
            end_time = time.time()
            
            if response.status_code == 200:
                audio_data = response.content
                duration = end_time - start_time
                file_size = len(audio_data)
                
                print(f"✅ TTS推理成功!")
                print(f"   耗时: {duration:.2f}秒")
                print(f"   文件大小: {file_size/1024:.1f}KB")
                
                # 保存音频文件
                if output_file:
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    print(f"   保存到: {output_file}")
                
                return audio_data
            else:
                print(f"❌ TTS推理失败: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return None


def main():
    """主函数 - 演示客户端使用"""
    print("🎤 TTS客户端示例")
    print("=" * 50)
    
    # 创建客户端
    client = TTSClient("http://219.144.21.182:9880")
    
    # 检查API健康状态
    if not client.check_health():
        print("❌ API服务未运行，请先启动: python tts_api.py")
        return
    
    print("✅ API服务正常运行")
    print()
    
    # 获取voice列表
    print("📋 获取可用voice列表...")
    voices_result = client.list_voices()
    voices = voices_result.get("voices", {})
    
    if not voices:
        print("❌ 没有找到可用的voice")
        return
    
    print(f"✅ 找到 {len(voices)} 个voice:")
    for voice_name, voice_info in voices.items():
        gender = voice_info.get("gender", "unknown")
        description = voice_info.get("description", "")
        audio_files = voice_info.get("audio_files", [])
        print(f"   {voice_name} ({gender}) - {description}")
        print(f"     音频文件: {', '.join(audio_files)}")
    print()
    
    # 演示使用voice进行TTS
    voice_name = list(voices.keys())[0]  # 使用第一个voice
    test_text = "你好，这是一个TTS测试，很高兴认识你！"
    
    print(f"🎵 使用voice '{voice_name}' 进行TTS测试...")
    output_file = f"test_output_{voice_name}.wav"
    
    # 使用voice进行TTS
    audio_data = client.tts_with_voice(
        voice_name=voice_name,
        text=test_text,
        output_file=output_file,
        temperature=1.0,
        speed_factor=1.0
    )
    
    if audio_data:
        print(f"✅ TTS测试完成，音频已保存到: {output_file}")
    else:
        print("❌ TTS测试失败")
    
    print()
    print("🎉 客户端示例演示完成!")


if __name__ == "__main__":
    main() 