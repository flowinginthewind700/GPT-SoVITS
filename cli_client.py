#!/usr/bin/env python3
"""
GPT-SoVITS命令行客户端
正确使用API的完整示例
"""
import requests
import json
import os
import sys
import argparse

class GPTSoVITSClient:
    def __init__(self, base_url="http://127.0.0.1:9880"):
        self.base_url = base_url.rstrip("/")
        
    def check_health(self):
        """检查API健康状态"""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def text_to_speech(self, text, ref_audio_path, text_lang="zh", prompt_lang="zh", prompt_text="", **kwargs):
        """文本转语音"""
        
        # 验证文件存在
        if not os.path.exists(ref_audio_path):
            return False, f"参考音频文件不存在: {ref_audio_path}"
        
        # 使用绝对路径
        ref_audio_path = os.path.abspath(ref_audio_path)
        
        # 构建payload
        payload = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_audio_path,
            "prompt_lang": prompt_lang.lower(),
            "prompt_text": prompt_text,
            "top_k": kwargs.get("top_k", 5),
            "top_p": kwargs.get("top_p", 1.0),
            "temperature": kwargs.get("temperature", 1.0),
            "text_split_method": kwargs.get("text_split_method", "cut5"),
            "batch_size": kwargs.get("batch_size", 1),
            "speed_factor": kwargs.get("speed_factor", 1.0),
            "seed": kwargs.get("seed", -1),
            "media_type": kwargs.get("media_type", "wav"),
            "streaming_mode": False,
            "parallel_infer": True
        }
        
        try:
            print("🚀 发送TTS请求...")
            response = requests.post(f"{self.base_url}/tts", json=payload, timeout=120)
            
            if response.status_code == 200:
                return True, response.content
            else:
                try:
                    error = response.json()
                    return False, error.get('message', str(error))
                except:
                    return False, response.text
                    
        except Exception as e:
            return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS命令行客户端")
    parser.add_argument("--text", required=True, help="要合成的文本")
    parser.add_argument("--ref-audio", required=True, help="参考音频文件路径")
    parser.add_argument("--text-lang", default="zh", help="文本语言 (默认: zh)")
    parser.add_argument("--prompt-lang", default="zh", help="提示语言 (默认: zh)")
    parser.add_argument("--prompt-text", default="", help="提示文本 (可选)")
    parser.add_argument("--output", default="output.wav", help="输出文件名 (默认: output.wav)")
    parser.add_argument("--server", default="http://127.0.0.1:9880", help="服务器地址")
    
    args = parser.parse_args()
    
    print("🎤 GPT-SoVITS命令行客户端")
    print("=" * 50)
    
    client = GPTSoVITSClient(args.server)
    
    # 检查API
    if not client.check_health():
        print("❌ API服务未运行，请先启动: python api_v2.py")
        return 1
    
    print("✅ API服务运行正常")
    
    # 检查参考音频
    if not os.path.exists(args.ref_audio):
        print(f"❌ 参考音频文件不存在: {args.ref_audio}")
        return 1
    
    print(f"📁 参考音频: {args.ref_audio}")
    print(f"📝 合成文本: {args.text}")
    print(f"🌐 文本语言: {args.text_lang}")
    print(f"🎯 提示语言: {args.prompt_lang}")
    
    # 执行TTS
    success, result = client.text_to_speech(
        text=args.text,
        ref_audio_path=args.ref_audio,
        text_lang=args.text_lang,
        prompt_lang=args.prompt_lang,
        prompt_text=args.prompt_text
    )
    
    if success:
        with open(args.output, "wb") as f:
            f.write(result)
        print(f"✅ 成功! 音频已保存: {args.output}")
        print(f"📊 文件大小: {len(result)/1024:.1f} KB")
    else:
        print(f"❌ 失败: {result}")
        return 1
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🎤 GPT-SoVITS命令行客户端")
        print("=" * 50)
        print("使用示例:")
        print("  python cli_client.py --text '你好世界' --ref-audio ./sample.wav")
        print("  python cli_client.py --text 'hello' --ref-audio ./ref.wav --text-lang en")
        print("\n参数说明:")
        print("  --text: 要合成的文本")
        print("  --ref-audio: 参考音频文件路径")
        print("  --text-lang: 文本语言 (zh/en/ja/ko)")
        print("  --output: 输出文件名")
        print("  --server: API服务器地址")
    else:
        sys.exit(main())