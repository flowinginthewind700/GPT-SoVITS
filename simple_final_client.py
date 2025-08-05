#!/usr/bin/env python3
"""
GPT-SoVITS 音色管理客户端
支持音色管理、缓存刷新、TTS推理等功能
"""
import gradio as gr
import requests
import os
import json
import time
from typing import Optional, Dict, Any

class VoiceManager:
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
    
    def get_voices(self) -> Dict[str, Any]:
        """获取所有可用的voice列表"""
        try:
            response = self.session.get(f"{self.base_url}/voices")
            if response.status_code == 200:
                return response.json()
            else:
                return {"voices": {}}
        except Exception as e:
            print(f"❌ 获取voice列表失败: {e}")
            return {"voices": {}}
    
    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态"""
        try:
            response = self.session.get(f"{self.base_url}/voice_cache_status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"cached_voices": [], "cache_size": 0}
        except Exception as e:
            print(f"❌ 获取缓存状态失败: {e}")
            return {"cached_voices": [], "cache_size": 0}
    
    def refresh_cache(self) -> Dict[str, Any]:
        """刷新音色缓存"""
        try:
            response = self.session.post(f"{self.base_url}/refresh_voice_cache")
            if response.status_code == 200:
                return response.json()
            else:
                return {"message": "刷新失败", "result": {}}
        except Exception as e:
            print(f"❌ 刷新缓存失败: {e}")
            return {"message": "刷新失败", "result": {}}
    
    def upload_voice(self, voice_name: str, audio_file, text_file=None, gender="unknown", description="", language="zh") -> Dict[str, Any]:
        """上传新音色"""
        try:
            files = {"audio_file": audio_file}
            if text_file:
                files["text_file"] = text_file
            
            data = {
                "voice_name": voice_name,
                "gender": gender,
                "description": description,
                "language": language
            }
            
            response = self.session.post(f"{self.base_url}/upload_voice", files=files, data=data)
            if response.status_code == 200:
                return response.json()
            else:
                return {"message": "上传失败", "error": response.text}
        except Exception as e:
            print(f"❌ 上传音色失败: {e}")
            return {"message": "上传失败", "error": str(e)}
    
    def delete_voice(self, voice_name: str) -> Dict[str, Any]:
        """删除音色"""
        try:
            response = self.session.delete(f"{self.base_url}/voice/{voice_name}")
            if response.status_code == 200:
                return response.json()
            else:
                return {"message": "删除失败", "error": response.text}
        except Exception as e:
            print(f"❌ 删除音色失败: {e}")
            return {"message": "删除失败", "error": str(e)}
    
    def tts_with_voice(self, voice_name: str, text: str, **kwargs) -> Optional[bytes]:
        """使用音色进行TTS推理"""
        try:
            params = {
                "voice_name": voice_name,
                "text": text,
                "text_lang": kwargs.get("text_lang", "zh"),
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
            
            response = self.session.post(f"{self.base_url}/tts_with_cached_voice", params=params, timeout=180)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"❌ TTS推理失败: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ TTS推理异常: {e}")
            return None

# 创建全局voice管理器
voice_manager = VoiceManager()

def check_api_status():
    """检查API状态"""
    if voice_manager.check_health():
        return "✅ API服务正常运行"
    else:
        return "❌ API服务未运行，请先启动: python tts_api.py"

def refresh_voice_cache():
    """刷新音色缓存"""
    result = voice_manager.refresh_cache()
    if result.get("message") == "success":
        cache_result = result.get("result", {})
        return f"✅ 缓存刷新成功!\n成功: {cache_result.get('success_count', 0)} 个\n失败: {cache_result.get('error_count', 0)} 个\n总计: {cache_result.get('total_voices', 0)} 个"
    else:
        return f"❌ 缓存刷新失败: {result.get('error', '未知错误')}"

def get_voice_list():
    """获取音色列表"""
    voices_result = voice_manager.get_voices()
    voices = voices_result.get("voices", {})
    
    if not voices:
        return "❌ 没有找到可用的音色"
    
    voice_list = []
    for voice_name, voice_info in voices.items():
        gender = voice_info.get("gender", "unknown")
        description = voice_info.get("description", "")
        audio_files = voice_info.get("audio_files", [])
        text_files = voice_info.get("text_files", [])
        
        voice_list.append(f"🎤 {voice_name} ({gender})")
        if description:
            voice_list.append(f"   描述: {description}")
        voice_list.append(f"   音频文件: {', '.join(audio_files)}")
        voice_list.append(f"   文本文件: {', '.join(text_files)}")
        voice_list.append("")
    
    return "\n".join(voice_list)

def get_cache_status():
    """获取缓存状态"""
    cache_result = voice_manager.get_cache_status()
    cached_voices = cache_result.get("cached_voices", [])
    cache_size = cache_result.get("cache_size", 0)
    cache_info = cache_result.get("cache_info", {})
    
    if not cached_voices:
        return "❌ 缓存为空，请先刷新缓存"
    
    status_list = [f"📊 缓存状态: {cache_size} 个音色已缓存\n"]
    for voice_name, info in cache_info.items():
        audio_size = info.get("audio_size", 0)
        prompt_text = info.get("prompt_text", "")
        gender = info.get("gender", "unknown")
        
        status_list.append(f"🎤 {voice_name} ({gender})")
        status_list.append(f"   音频大小: {audio_size/1024:.1f}KB")
        status_list.append(f"   提示文本: {prompt_text}")
        status_list.append("")
    
    return "\n".join(status_list)

def upload_new_voice(voice_name, audio_file, text_file, gender, description, language):
    """上传新音色"""
    if not voice_name or not voice_name.strip():
        return "❌ 请输入音色名称"
    
    if not audio_file:
        return "❌ 请选择音频文件"
    
    result = voice_manager.upload_voice(
        voice_name=voice_name.strip(),
        audio_file=audio_file,
        text_file=text_file,
        gender=gender,
        description=description,
        language=language
    )
    
    if result.get("message") == "success":
        return f"✅ 音色上传成功!\n音色名称: {result.get('voice_name')}\n音频文件: {result.get('audio_file')}\n文本文件: {result.get('text_file')}\n缓存加载: {'成功' if result.get('cache_loaded') else '失败'}"
    else:
        return f"❌ 音色上传失败: {result.get('error', '未知错误')}"

def delete_voice_by_name(voice_name):
    """删除音色"""
    if not voice_name or not voice_name.strip():
        return "❌ 请输入要删除的音色名称"
    
    result = voice_manager.delete_voice(voice_name.strip())
    
    if result.get("message") == "success":
        return f"✅ 音色删除成功: {voice_name}"
    else:
        return f"❌ 音色删除失败: {result.get('error', '未知错误')}"

def tts_with_selected_voice(voice_name, text, text_lang, temperature, speed_factor):
    """使用选定的音色进行TTS"""
    if not voice_name or not voice_name.strip():
        return None, "❌ 请选择音色"
    
    if not text or not text.strip():
        return None, "❌ 请输入要合成的文本"
    
    print(f"🎵 开始TTS推理...")
    print(f"   音色: {voice_name}")
    print(f"   文本: {text}")
    
    start_time = time.time()
    audio_data = voice_manager.tts_with_voice(
        voice_name=voice_name.strip(),
        text=text.strip(),
        text_lang=text_lang,
        temperature=temperature,
        speed_factor=speed_factor
    )
    end_time = time.time()
    
    if audio_data:
        # 保存音频文件
        output_filename = f"tts_output_{voice_name}_{int(time.time())}.wav"
        with open(output_filename, "wb") as f:
            f.write(audio_data)
        
        duration = end_time - start_time
        file_size = len(audio_data)
        
        success_msg = f"✅ TTS推理成功!\n耗时: {duration:.2f}秒\n文件大小: {file_size/1024:.1f}KB\n保存到: {output_filename}"
        
        return output_filename, success_msg
    else:
        return None, "❌ TTS推理失败"

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="GPT-SoVITS 音色管理客户端") as app:
        gr.Markdown("# 🎤 GPT-SoVITS 音色管理客户端")
        
        # API状态
        with gr.Row():
            api_status = gr.Textbox(
                label="API状态",
                value=check_api_status(),
                interactive=False
            )
            refresh_status_btn = gr.Button("🔄 刷新状态", size="sm")
        
        # 音色管理标签页
        with gr.Tabs():
            # 音色列表和缓存管理
            with gr.Tab("📋 音色管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 缓存管理")
                        refresh_cache_btn = gr.Button("🔄 刷新音色缓存", variant="primary")
                        cache_status_text = gr.Textbox(
                            label="缓存状态",
                            value=get_cache_status(),
                            lines=10,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 音色列表")
                        refresh_voices_btn = gr.Button("🔄 刷新音色列表", variant="primary")
                        voice_list_text = gr.Textbox(
                            label="可用音色",
                            value=get_voice_list(),
                            lines=10,
                            interactive=False
                        )
            
            # 音色上传
            with gr.Tab("📤 音色上传"):
                gr.Markdown("### 上传新音色")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        voice_name_input = gr.Textbox(
                            label="音色名称",
                            placeholder="请输入音色名称...",
                            lines=1
                        )
                        
                        audio_file_input = gr.File(
                            label="音频文件",
                            file_types=["audio"]
                        )
                        
                        text_file_input = gr.File(
                            label="文本文件 (可选)",
                            file_types=["text"]
                        )
                    
                    with gr.Column(scale=1):
                        gender_select = gr.Dropdown(
                            choices=["unknown", "male", "female"],
                            value="unknown",
                            label="性别"
                        )
                        
                        description_input = gr.Textbox(
                            label="描述",
                            placeholder="音色描述...",
                            lines=2
                        )
                        
                        language_select = gr.Dropdown(
                            choices=["zh", "en", "ja", "ko"],
                            value="zh",
                            label="语言"
                        )
                
                upload_btn = gr.Button("📤 上传音色", variant="primary")
                upload_result = gr.Textbox(
                    label="上传结果",
                    interactive=False,
                    lines=3
                )
            
            # 音色删除
            with gr.Tab("🗑️ 音色删除"):
                gr.Markdown("### 删除音色")
                
                delete_voice_input = gr.Textbox(
                    label="音色名称",
                    placeholder="请输入要删除的音色名称...",
                    lines=1
                )
                
                delete_btn = gr.Button("🗑️ 删除音色", variant="stop")
                delete_result = gr.Textbox(
                    label="删除结果",
                    interactive=False,
                    lines=2
                )
            
            # TTS推理
            with gr.Tab("🎵 TTS推理"):
                gr.Markdown("### 使用音色进行TTS推理")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # 获取可用音色列表
                        voices_result = voice_manager.get_voices()
                        voices = voices_result.get("voices", {})
                        voice_choices = list(voices.keys()) if voices else []
                        
                        voice_select = gr.Dropdown(
                            choices=voice_choices,
                            label="选择音色",
                            value=voice_choices[0] if voice_choices else None
                        )
                        
                        tts_text_input = gr.Textbox(
                            label="要合成的文本",
                            placeholder="请输入要合成的文本...",
                            lines=4
                        )
                        
                        text_lang_select = gr.Dropdown(
                            choices=["zh", "en", "auto"],
                            value="zh",
                            label="文本语言"
                        )
                    
                    with gr.Column(scale=1):
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature"
                        )
                        
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed Factor"
                        )
                        
                        tts_btn = gr.Button("🎵 开始TTS", variant="primary", size="lg")
                
                with gr.Row():
                    tts_output_audio = gr.Audio(label="生成的音频")
                    tts_status = gr.Textbox(label="状态信息", interactive=False)
        
        # 事件绑定
        refresh_status_btn.click(
            fn=check_api_status,
            outputs=api_status
        )
        
        refresh_cache_btn.click(
            fn=refresh_voice_cache,
            outputs=cache_status_text
        )
        
        refresh_voices_btn.click(
            fn=get_voice_list,
            outputs=voice_list_text
        )
        
        upload_btn.click(
            fn=upload_new_voice,
            inputs=[voice_name_input, audio_file_input, text_file_input, gender_select, description_input, language_select],
            outputs=upload_result
        )
        
        delete_btn.click(
            fn=delete_voice_by_name,
            inputs=delete_voice_input,
            outputs=delete_result
        )
        
        tts_btn.click(
            fn=tts_with_selected_voice,
            inputs=[voice_select, tts_text_input, text_lang_select, temperature_slider, speed_slider],
            outputs=[tts_output_audio, tts_status]
        )
        
        # 使用说明
        gr.Markdown("""
        ### 📖 使用说明
        
        #### 🚀 快速开始
        1. **检查API状态**: 确保API服务正常运行
        2. **刷新缓存**: 点击"刷新音色缓存"加载所有音色到内存
        3. **选择音色**: 在TTS推理标签页选择要使用的音色
        4. **开始推理**: 输入文本并点击"开始TTS"
        
        #### 📤 添加新音色
        1. 准备音频文件（支持wav、mp3、flac等格式）
        2. 准备对应的文本文件（可选）
        3. 填写音色信息并上传
        
        #### 🗑️ 删除音色
        1. 输入要删除的音色名称
        2. 点击删除按钮（操作不可逆）
        
        ### ⚠️ 注意事项
        - 音色名称不能包含特殊字符
        - 音频文件建议时长3-10秒
        - 文本文件使用UTF-8编码
        - 删除操作不可恢复
        """)
    
    return app

if __name__ == "__main__":
    print("🚀 启动GPT-SoVITS 音色管理客户端...")
    print("📋 请确保已运行: python tts_api.py")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        inbrowser=True,
        show_error=True
    ) 