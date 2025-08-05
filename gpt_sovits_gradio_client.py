import gradio as gr
import requests
import json
import os
import tempfile
import numpy as np
import soundfile as sf
from io import BytesIO
from typing import Tuple, Optional

class GPTSoVITSClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9880"):
        self.base_url = base_url.rstrip("/")
        
    def tts(self, 
            text: str,
            text_lang: str,
            ref_audio_path: str,
            prompt_lang: str,
            prompt_text: str = "",
            top_k: int = 5,
            top_p: float = 1.0,
            temperature: float = 1.0,
            text_split_method: str = "cut5",
            batch_size: int = 1,
            speed_factor: float = 1.0,
            seed: int = -1,
            media_type: str = "wav",
            streaming_mode: bool = False) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        调用TTS API生成语音
        
        返回: (audio_data, sample_rate) 或 (None, None) 如果失败
        """
        try:
            # 如果参考音频是上传的文件，使用临时路径
            if hasattr(ref_audio_path, 'name'):
                ref_audio_path = ref_audio_path.name
            
            payload = {
                "text": text,
                "text_lang": text_lang.lower(),
                "ref_audio_path": ref_audio_path,
                "prompt_lang": prompt_lang.lower(),
                "prompt_text": prompt_text,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "text_split_method": text_split_method,
                "batch_size": batch_size,
                "speed_factor": speed_factor,
                "seed": seed,
                "media_type": media_type,
                "streaming_mode": streaming_mode
            }
            
            response = requests.post(f"{self.base_url}/tts", json=payload, timeout=300)
            
            if response.status_code == 200:
                # 将音频数据加载为numpy数组
                audio_data, sample_rate = sf.read(BytesIO(response.content))
                return audio_data, sample_rate
            else:
                print(f"TTS API错误: {response.status_code} - {response.text}")
                return None, None
                
        except Exception as e:
            print(f"调用TTS API时出错: {str(e)}")
            return None, None
    
    def set_gpt_weights(self, weights_path: str) -> bool:
        """设置GPT模型权重"""
        try:
            response = requests.get(f"{self.base_url}/set_gpt_weights", 
                                  params={"weights_path": weights_path})
            return response.status_code == 200
        except Exception as e:
            print(f"设置GPT权重时出错: {str(e)}")
            return False
    
    def set_sovits_weights(self, weights_path: str) -> bool:
        """设置SoVITS模型权重"""
        try:
            response = requests.get(f"{self.base_url}/set_sovits_weights", 
                                  params={"weights_path": weights_path})
            return response.status_code == 200
        except Exception as e:
            print(f"设置SoVITS权重时出错: {str(e)}")
            return False

# 全局客户端实例
client = GPTSoVITSClient()

# 支持的语言
LANGUAGES = ["zh", "en", "ja", "ko", "yue", "minnan", "zh-yue", "ja-jp", "en-us", "ko-kr"]
TEXT_SPLIT_METHODS = ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5", "cut_custom"]

# 预训练模型路径（示例）
PRETRAINED_MODELS = {
    "GPT模型": [
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    ],
    "SoVITS模型": [
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G488k.pth"
    ]
}

def tts_function(text, text_lang, ref_audio, prompt_text, prompt_lang, 
                top_k, top_p, temperature, text_split_method, batch_size, 
                speed_factor, seed, media_type):
    """Gradio接口函数"""
    if not text:
        return None, "错误：请输入要合成的文本"
    
    if not ref_audio:
        return None, "错误：请上传参考音频文件"
    
    audio_data, sample_rate = client.tts(
        text=text,
        text_lang=text_lang,
        ref_audio_path=ref_audio,
        prompt_lang=prompt_lang,
        prompt_text=prompt_text,
        top_k=int(top_k),
        top_p=float(top_p),
        temperature=float(temperature),
        text_split_method=text_split_method,
        batch_size=int(batch_size),
        speed_factor=float(speed_factor),
        seed=int(seed),
        media_type=media_type
    )
    
    if audio_data is not None:
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix=f".{media_type}", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            return tmp_file.name, "成功：语音合成完成"
    else:
        return None, "错误：语音合成失败"

def load_models_tab():
    """模型管理标签页"""
    with gr.Tab("模型管理"):
        gr.Markdown("## 模型权重管理")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### GPT模型")
                gpt_weights_path = gr.Textbox(
                    label="GPT权重路径",
                    value=PRETRAINED_MODELS["GPT模型"][0],
                    placeholder="输入GPT权重文件路径"
                )
                load_gpt_btn = gr.Button("加载GPT模型")
                gpt_status = gr.Textbox(label="GPT状态", interactive=False)
                
            with gr.Column():
                gr.Markdown("### SoVITS模型")
                sovits_weights_path = gr.Textbox(
                    label="SoVITS权重路径",
                    value=PRETRAINED_MODELS["SoVITS模型"][0],
                    placeholder="输入SoVITS权重文件路径"
                )
                load_sovits_btn = gr.Button("加载SoVITS模型")
                sovits_status = gr.Textbox(label="SoVITS状态", interactive=False)
        
        def load_gpt_model(weights_path):
            if client.set_gpt_weights(weights_path):
                return f"成功加载GPT模型: {os.path.basename(weights_path)}"
            else:
                return f"加载GPT模型失败: {os.path.basename(weights_path)}"
        
        def load_sovits_model(weights_path):
            if client.set_sovits_weights(weights_path):
                return f"成功加载SoVITS模型: {os.path.basename(weights_path)}"
            else:
                return f"加载SoVITS模型失败: {os.path.basename(weights_path)}"
        
        load_gpt_btn.click(load_gpt_model, inputs=[gpt_weights_path], outputs=[gpt_status])
        load_sovits_btn.click(load_sovits_model, inputs=[sovits_weights_path], outputs=[sovits_status])

def tts_tab():
    """TTS标签页"""
    with gr.Tab("文本转语音"):
        gr.Markdown("## GPT-SoVITS 文本转语音")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="要合成的文本",
                    placeholder="请输入要合成的文本内容...",
                    lines=4,
                    max_lines=10
                )
                
                with gr.Row():
                    text_lang = gr.Dropdown(
                        choices=LANGUAGES,
                        value="zh",
                        label="文本语言"
                    )
                    prompt_lang = gr.Dropdown(
                        choices=LANGUAGES,
                        value="zh",
                        label="提示文本语言"
                    )
                
                prompt_text = gr.Textbox(
                    label="提示文本",
                    placeholder="请输入参考音频对应的文本内容...",
                    lines=2
                )
                
                ref_audio = gr.Audio(
                    label="参考音频",
                    type="filepath",
                    sources=["upload"]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                
                with gr.Row():
                    top_k = gr.Slider(1, 50, value=5, step=1, label="Top K")
                    top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Top P")
                
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="温度")
                speed_factor = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
                batch_size = gr.Slider(1, 10, value=1, step=1, label="批大小")
                
                text_split_method = gr.Dropdown(
                    choices=TEXT_SPLIT_METHODS,
                    value="cut5",
                    label="文本切分方法"
                )
                
                seed = gr.Number(value=-1, label="随机种子(-1为随机)")
                media_type = gr.Radio(
                    choices=["wav", "mp3", "ogg"],
                    value="wav",
                    label="输出格式"
                )
        
        generate_btn = gr.Button("生成语音", variant="primary")
        
        with gr.Row():
            audio_output = gr.Audio(label="生成的语音")
            status_output = gr.Textbox(label="状态", interactive=False)
        
        # 绑定生成函数
        generate_btn.click(
            fn=tts_function,
            inputs=[
                text_input, text_lang, ref_audio, prompt_text, prompt_lang,
                top_k, top_p, temperature, text_split_method, batch_size,
                speed_factor, seed, media_type
            ],
            outputs=[audio_output, status_output]
        )

# 创建主界面
def create_interface():
    with gr.Blocks(title="GPT-SoVITS 客户端", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# GPT-SoVITS 语音合成客户端")
        gr.Markdown("基于GPT-SoVITS的文本转语音系统")
        
        tts_tab()
        load_models_tab()
        
        gr.Markdown("---")
        gr.Markdown("### 使用说明")
        gr.Markdown("""
        1. **上传参考音频**: 选择一段参考音频作为音色模板
        2. **输入提示文本**: 输入参考音频对应的文本内容（可选）
        3. **设置参数**: 调整各种合成参数
        4. **生成语音**: 点击生成按钮开始合成
        5. **模型管理**: 在模型管理标签页中切换不同的模型权重
        """)
    
    return interface

if __name__ == "__main__":
    import os
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )