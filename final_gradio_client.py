#!/usr/bin/env python3
"""
最终版GPT-SoVITS Gradio客户端
包含错误处理和用户指导
"""
import gradio as gr
import requests
import os
import json
import re

def check_api_health():
    """检查API健康状态"""
    try:
        response = requests.get("http://219.144.21.182:9880/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

def detect_language(text):
    """检测文本主要语言"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if chinese_chars > english_chars:
        return "zh"
    elif english_chars > chinese_chars:
        return "en"
    else:
        return "zh"  # 默认中文

def split_mixed_text(text):
    """智能分割中英文混合文本"""
    # 如果文本主要是单一语言，直接返回
    if is_pure_language(text):
        return [text]
    
    # 使用正则表达式分割中英文
    pattern = r'([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+|[a-zA-Z0-9\s.,!?;:()"\'-]+)'
    matches = re.findall(pattern, text)
    
    segments = []
    for segment in matches:
        segment = segment.strip()
        if segment and len(segment) > 1:  # 过滤掉太短的片段
            segments.append(segment)
    
    return segments if segments else [text]

def is_pure_language(text):
    """判断是否为单一语言文本"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # 如果一种语言占90%以上，认为是单一语言
    total_chars = chinese_chars + english_chars
    if total_chars == 0:
        return True
    
    return chinese_chars / total_chars > 0.9 or english_chars / total_chars > 0.9

def split_long_text(text, max_length=200):
    """分割过长的文本"""
    if len(text) <= max_length:
        return [text]
    
    # 使用标点符号分割
    punctuation = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    segments = []
    current_segment = ""
    
    for char in text:
        current_segment += char
        if char in punctuation and len(current_segment) >= max_length // 2:
            segments.append(current_segment.strip())
            current_segment = ""
    
    if current_segment.strip():
        segments.append(current_segment.strip())
    
    return segments if segments else [text]

def preprocess_mixed_text(text):
    """预处理中英文混合文本"""
    # 检测是否为混合文本
    if not is_pure_language(text):
        print("🔍 检测到中英文混合文本，进行预处理...")
        # 对于混合文本，使用更保守的分割方法
        return text, "cut0"  # 不分割，让服务端处理
    
    # 单一语言文本，使用正常分割
    if len(text) > 200:
        return text, "cut2"  # 按50字分割
    else:
        return text, "cut5"  # 按标点符号分割

def get_gpt_sovits_language(text):
    """获取GPT-SoVITS支持的语言参数"""
    detected_lang = detect_language(text)
    is_mixed = not is_pure_language(text)
    
    if is_mixed:
        # 多语言混合，使用auto
        return "auto"
    elif detected_lang == "zh":
        # 纯中文，使用zh（v4支持）
        return "zh"
    elif detected_lang == "en":
        # 纯英文，使用en
        return "en"
    elif detected_lang == "ja":
        # 纯日文，使用ja
        return "ja"
    elif detected_lang == "ko":
        # 纯韩文，使用ko
        return "ko"
    else:
        # 其他情况，使用auto
        return "auto"

def get_text_language_advice(text, ref_audio_lang="zh"):
    """获取文本语言建议"""
    detected_lang = detect_language(text)
    is_mixed = not is_pure_language(text)
    
    if is_mixed:
        if ref_audio_lang == "zh":
            return "auto", "⚠️ 检测到中英文混合文本，但参考音频为中文。建议：\n1. 使用纯中文文本\n2. 或准备中英文混合的参考音频"
        elif ref_audio_lang == "en":
            return "auto", "⚠️ 检测到中英文混合文本，但参考音频为英文。建议：\n1. 使用纯英文文本\n2. 或准备中英文混合的参考音频"
        else:
            return "auto", "⚠️ 检测到中英文混合文本，使用auto语言设置"
    else:
        if detected_lang == ref_audio_lang:
            return detected_lang, "✅ 文本语言与参考音频语言匹配"
        else:
            return detected_lang, f"⚠️ 文本语言({detected_lang})与参考音频语言({ref_audio_lang})不匹配"

def tts_with_error_handling(text, ref_audio, text_lang="zh", prompt_text="", prompt_lang="zh"):
    """带错误处理的TTS函数"""
    
    # 检查API
    if not check_api_health():
        return None, "❌ GPT-SoVITS API未运行，请先启动: python api_v2.py"
    
    # 验证输入
    if not text or not text.strip():
        return None, "❌ 请输入要合成的文本"
    
    if not ref_audio:
        return None, "❌ 请上传参考音频文件"
    
    # 处理文件路径
    if hasattr(ref_audio, 'name'):
        ref_audio_path = ref_audio.name
    else:
        ref_audio_path = str(ref_audio)
    
    if not os.path.exists(ref_audio_path):
        return None, f"❌ 参考音频文件不存在: {ref_audio_path}"
    
    # 使用绝对路径
    ref_audio_path = os.path.abspath(ref_audio_path)
    
    # 上传文件到服务器
    try:
        print("📤 上传音频文件到服务器...")
        with open(ref_audio_path, "rb") as f:
            files = {"audio_file": (os.path.basename(ref_audio_path), f, "audio/wav")}
            upload_response = requests.post("http://219.144.21.182:9880/set_refer_audio", files=files, timeout=30)
        
        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            if upload_result.get("message") == "success":
                print("✅ 文件上传成功")
                # 使用上传后的文件路径
                ref_audio_path = f"uploaded_audio/{os.path.basename(ref_audio_path)}"
            else:
                print(f"❌ 文件上传失败: {upload_result}")
                return None, f"❌ 文件上传失败: {upload_result}"
        else:
            print(f"❌ 文件上传失败: {upload_response.text}")
            return None, f"❌ 文件上传失败: {upload_response.text}"
    except Exception as e:
        print(f"❌ 文件上传异常: {e}")
        return None, f"❌ 文件上传异常: {e}"
    
    # 预处理文本
    processed_text, split_method = preprocess_mixed_text(text.strip())
    
    # 获取GPT-SoVITS支持的语言参数
    gpt_sovits_lang = get_gpt_sovits_language(processed_text)
    
    # 自动检测语言（如果用户选择auto）
    if text_lang == "auto":
        text_lang = gpt_sovits_lang
        print(f"🔍 自动检测语言: {text_lang}")
    
    if prompt_lang == "auto":
        detected_prompt_lang = detect_language(prompt_text) if prompt_text else text_lang
        prompt_lang = detected_prompt_lang
        print(f"🔍 自动检测提示语言: {detected_prompt_lang}")
    
    # 对于多语言混合文本，给出警告
    if not is_pure_language(processed_text):
        print("🔄 检测到多语言混合文本，使用auto语言设置")
        text_lang = "auto"
        prompt_lang = "auto"
        warning_msg = "⚠️ 多语言混合文本可能影响音色一致性，建议使用纯语言文本"
        print(warning_msg)
    
    # 构建请求
    payload = {
        "text": processed_text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "prompt_lang": prompt_lang.lower(),
        "prompt_text": prompt_text.strip(),
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": split_method,
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": False,  # v4模型不支持bucket处理
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": -1,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False
    }
    
    try:
        print("🚀 发送TTS请求...")
        print(f"📊 请求参数: text_lang={text_lang}, prompt_lang={prompt_lang}, split_method={split_method}")
        print(f"📊 完整payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        response = requests.post("http://219.144.21.182:9880/tts", json=payload, timeout=180)  # 增加超时时间
        
        if response.status_code == 200:
            # 保存结果
            output_path = "tts_output.wav"
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            file_size = len(response.content)
            duration_estimate = file_size / (16000 * 2)  # 粗略估计
            
            success_msg = f"✅ 生成成功! 文件大小: {file_size/1024:.1f}KB, 估计时长: {duration_estimate:.1f}s"
            if not is_pure_language(processed_text):
                success_msg += "\n⚠️ 混合文本可能影响音色一致性"
            
            return output_path, success_msg
            
        else:
            error_msg = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get('message', str(error_json))
            except:
                pass
            
            # 如果是400错误，尝试不同的语言设置
            if response.status_code == 400 and "failed" in error_msg.lower():
                print("🔄 尝试不同的语言设置...")
                print(f"📄 原始错误: {error_msg}")
                for alt_lang in ["zh", "all_zh", "en", "auto"]:
                    if alt_lang != text_lang:
                        print(f"🔄 尝试语言: {alt_lang}")
                        # 确保所有必要的参数都存在
                        retry_payload = payload.copy()
                        retry_payload["text_lang"] = alt_lang
                        retry_payload["prompt_lang"] = alt_lang
                        retry_payload["split_bucket"] = False  # 确保v4兼容
                        
                        print(f"📊 重试参数: text_lang={alt_lang}, prompt_lang={alt_lang}")
                        alt_response = requests.post("http://219.144.21.182:9880/tts", json=retry_payload, timeout=120)
                        
                        print(f"📡 重试响应: {alt_response.status_code}")
                        if alt_response.status_code != 200:
                            try:
                                alt_error = alt_response.json()
                                print(f"📄 重试错误: {alt_error}")
                            except:
                                print(f"📄 重试错误: {alt_response.text}")
                        
                        if alt_response.status_code == 200:
                            output_path = "tts_output.wav"
                            with open(output_path, "wb") as f:
                                f.write(alt_response.content)
                            file_size = len(alt_response.content)
                            duration_estimate = file_size / (16000 * 2)
                            success_msg = f"✅ 生成成功! (使用{alt_lang}语言) 文件大小: {file_size/1024:.1f}KB, 估计时长: {duration_estimate:.1f}s"
                            if not is_pure_language(processed_text):
                                success_msg += "\n⚠️ 混合文本可能影响音色一致性"
                            return output_path, success_msg
            
            return None, f"❌ API错误: {response.status_code} - {error_msg}"
            
    except requests.exceptions.Timeout:
        return None, "❌ 请求超时，请检查音频文件大小或尝试更短的文本"
    except requests.exceptions.ConnectionError:
        return None, "❌ 连接失败，请确保API服务正在运行"
    except Exception as e:
        return None, f"❌ 请求异常: {str(e)}"

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="GPT-SoVITS 语音合成", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎤 GPT-SoVITS 语音合成客户端")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 输入内容")
                text_input = gr.Textbox(
                    label="要合成的文本",
                    placeholder="请输入要合成的文本内容...",
                    lines=4,
                    max_lines=8
                )
                
                prompt_text = gr.Textbox(
                    label="提示文本（可选）",
                    placeholder="输入参考音频对应的文本内容...",
                    lines=2
                )
                
                ref_audio = gr.Audio(
                    label="参考音频文件",
                    type="filepath",
                    sources=["upload"]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 语言设置")
                text_lang = gr.Dropdown(
                    choices=["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"],
                    value="auto",
                    label="文本语言"
                )
                
                prompt_lang = gr.Dropdown(
                    choices=["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"],
                    value="auto",
                    label="提示语言"
                )
                
                gr.Markdown("### 状态")
                api_status = gr.Textbox(
                    label="API状态",
                    value="检查中..." if check_api_health() else "❌ API未运行",
                    interactive=False
                )
        
        with gr.Row():
            generate_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
        
        with gr.Row():
            output_audio = gr.Audio(label="生成的语音文件")
            status_text = gr.Textbox(label="状态信息", interactive=False)
        
        # 事件绑定
        generate_btn.click(
            fn=tts_with_error_handling,
            inputs=[text_input, ref_audio, text_lang, prompt_text, prompt_lang],
            outputs=[output_audio, status_text]
        )
        
        # 添加使用说明
        gr.Markdown("""
        ### 📖 使用说明
        1. **启动API服务**: 在终端运行 `python api_v2.py`
        2. **上传参考音频**: 选择一段.wav格式的音频作为音色模板
        3. **输入文本**: 输入要合成的文本内容
        4. **生成语音**: 点击"生成语音"按钮
        
        ### ⚠️ 重要注意事项
        - **参考音频语言**: 建议参考音频的语言与合成文本语言一致
        - **中英文混合**: 如果参考音频是中文，建议使用纯中文文本
        - **音色一致性**: 混合语言可能影响音色的一致性
        - **参考音频质量**: 应为.wav格式，时长建议3-10秒
        
        ### 🔧 故障排除
        - 如果中英文混合出错，尝试使用纯中文或纯英文文本
        - 如果长句报错，系统会自动分割文本重试
        - 确保参考音频文件有效且格式正确
        - 混合文本会自动使用"auto"语言设置
        
        ### 💡 最佳实践
        - 中文参考音频 → 中文文本
        - 英文参考音频 → 英文文本
        - 混合参考音频 → 混合文本（如果可用）
        
        ### 🌐 语言支持
        - **auto**: 自动检测语言（推荐）
        - **zh**: 按中英混合识别
        - **en**: 全部按英文处理
        - **ja**: 按日英混合识别
        - **yue**: 按粤英混合识别
        - **ko**: 按韩英混合识别
        - **all_zh**: 全部按中文处理
        - **all_ja**: 全部按日文处理
        - **all_yue**: 全部按粤语处理
        - **all_ko**: 全部按韩文处理
        - **auto_yue**: 多语种启动切分识别语种（粤语）
        """)
    
    return app

if __name__ == "__main__":
    print("🚀 启动GPT-SoVITS Gradio客户端...")
    print("📋 请确保已运行: python api_v2.py")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True
    )