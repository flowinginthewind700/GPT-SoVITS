"""
# TTS API with Voice Management

` python tts_api.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False       # bool. whether to use super-sampling for audio when using VITS model V3.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import os
import sys
import traceback
import json
from typing import Generator

# 在文件开头添加性能监控
import time
import psutil
import threading
from functools import wraps

# 检查CUDA优化模块可用性
CUDA_OPTIMIZATION_AVAILABLE = True
try:
    import torch
    if not torch.cuda.is_available():
        CUDA_OPTIMIZATION_AVAILABLE = False
        print("⚠️ CUDA不可用，将使用原始Vocoder处理")
    else:
        print("✅ CUDA优化环境检查通过")
except ImportError:
    CUDA_OPTIMIZATION_AVAILABLE = False
    print("⚠️ PyTorch不可用，将使用原始Vocoder处理")

# 轻量级性能监控装饰器
def performance_monitor(func_name="", enable_memory_tracking=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if enable_memory_tracking:
                try:
                    memory_info = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    print(f"⏱️ {func_name or func.__name__}: {duration:.3f}s, 内存: {memory_info:.1f}MB")
                except:
                    print(f"⏱️ {func_name or func.__name__}: {duration:.3f}s")
            else:
                print(f"⏱️ {func_name or func.__name__}: {duration:.3f}s")
            
            return result
        return wrapper
    return decorator

# 全局性能统计
performance_stats = {
    "total_requests": 0,
    "total_tts_time": 0,
    "total_t2s_time": 0,
    "total_vocoder_time": 0,
    "total_postprocess_time": 0
}

def update_performance_stats(stage, duration):
    """更新性能统计"""
    performance_stats["total_requests"] += 1
    if stage == "t2s":
        performance_stats["total_t2s_time"] += duration
    elif stage == "vocoder":
        performance_stats["total_vocoder_time"] += duration
    elif stage == "postprocess":
        performance_stats["total_postprocess_time"] += duration
    elif stage == "total":
        performance_stats["total_tts_time"] += duration


now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel

# 导入CUDA优化模块
import torch
from typing import List

# 导入并发处理模块
try:
    from concurrent_tts_enhancement import (
        initialize_concurrent_tts, 
        shutdown_concurrent_tts,
        process_concurrent_tts,
        get_concurrent_stats
    )
    CONCURRENT_TTS_AVAILABLE = True
    print("✅ 并发TTS模块导入成功")
except ImportError as e:
    CONCURRENT_TTS_AVAILABLE = False
    print(f"⚠️ 并发TTS模块导入失败: {e}")
    print("将使用单线程处理")

class CUDAGraphVocoder:
    """高性能CUDA Graph优化的Vocoder"""
    
    def __init__(self, vocoder_model, device="cuda", max_cached_shapes=3):
        self.vocoder_model = vocoder_model
        self.device = device
        self.max_cached_shapes = max_cached_shapes
        self.cached_graphs = {}  # 缓存多个形状的graph
        self.warmup_done = False
        
    def initialize_graph(self, input_shape: tuple, num_warmup: int = 3):
        """初始化CUDA Graph - 支持多形状缓存"""
        if input_shape in self.cached_graphs:
            return True
            
        # 限制缓存数量，清理最老的
        if len(self.cached_graphs) >= self.max_cached_shapes:
            oldest_shape = next(iter(self.cached_graphs))
            del self.cached_graphs[oldest_shape]
            print(f"🔄 清理旧的CUDA Graph缓存: {oldest_shape}")
            
        print(f"🔄 初始化CUDA Graph (形状: {input_shape})...")
        
        try:
            # 全局预热(只需要一次)
            if not self.warmup_done:
                dummy_input = torch.randn(input_shape, device=self.device, dtype=torch.float16)
                for _ in range(num_warmup):
                    with torch.no_grad():
                        _ = self.vocoder_model(dummy_input)
                torch.cuda.synchronize()
                self.warmup_done = True
            
            # 创建静态输入输出
            static_input = torch.randn(input_shape, device=self.device, dtype=torch.float16)
            
            # 捕获CUDA Graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    static_output = self.vocoder_model(static_input)
            
            # 缓存graph信息
            self.cached_graphs[input_shape] = {
                'graph': graph,
                'static_input': static_input,
                'static_output': static_output
            }
            
            print(f"✅ CUDA Graph初始化完成 (缓存数: {len(self.cached_graphs)})")
            return True
            
        except Exception as e:
            print(f"❌ CUDA Graph初始化失败: {e}")
            return False
    
    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """使用CUDA Graph进行推理"""
        input_shape = tuple(input_tensor.shape)
        
        # 检查缓存
        if input_shape not in self.cached_graphs:
            if not self.initialize_graph(input_shape):
                # 回退到普通推理
                with torch.no_grad():
                    return self.vocoder_model(input_tensor)
        
        # 使用缓存的graph
        graph_info = self.cached_graphs[input_shape]
        graph_info['static_input'].copy_(input_tensor)
        graph_info['graph'].replay()
        
        return graph_info['static_output'].clone()

class OptimizedVocoderProcessor:
    """高性能Vocoder处理器"""
    
    def __init__(self, vocoder_model, device="cuda", chunk_size=500, overlap_ratio=0.1):
        self.vocoder_model = vocoder_model
        self.device = device
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.cuda_graph_vocoder = CUDAGraphVocoder(vocoder_model, device)
        self.performance_history = []
        
    def get_optimal_chunk_size(self, total_length: int) -> int:
        """根据历史性能数据动态调整chunk_size"""
        if len(self.performance_history) < 3:
            return self.chunk_size
            
        # 简单的自适应策略
        avg_time_per_chunk = sum(self.performance_history[-3:]) / 3
        if avg_time_per_chunk > 2.0:  # 如果处理时间过长
            return max(300, self.chunk_size - 100)
        elif avg_time_per_chunk < 0.5:  # 如果处理很快
            return min(800, self.chunk_size + 100)
        return self.chunk_size
    
    def split_into_chunks(self, pred_spec: torch.Tensor) -> List[torch.Tensor]:
        """智能分割为块，支持重叠处理"""
        total_length = pred_spec.shape[-1]
        optimal_chunk_size = self.get_optimal_chunk_size(total_length)
        
        if total_length <= optimal_chunk_size:
            return [pred_spec]
        
        chunks = []
        overlap_size = int(optimal_chunk_size * self.overlap_ratio)
        step_size = optimal_chunk_size - overlap_size
        
        for start in range(0, total_length, step_size):
            end = min(start + optimal_chunk_size, total_length)
            chunk = pred_spec[:, :, start:end]
            chunks.append(chunk)
            
            if end >= total_length:
                break
        
        return chunks
    
    def merge_overlapped_chunks(self, audio_chunks: List[torch.Tensor]) -> torch.Tensor:
        """合并重叠的音频块，使用窗函数平滑"""
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # 简化版本：直接拼接，忽略重叠部分的平滑处理
        # 在生产环境中可以添加更复杂的平滑算法
        total_length = sum(chunk.shape[-1] for chunk in audio_chunks)
        overlap_size = int(self.chunk_size * self.overlap_ratio)
        adjusted_length = total_length - (len(audio_chunks) - 1) * overlap_size
        
        return torch.cat(audio_chunks, dim=-1)[:, :adjusted_length]
    
    def process_optimized(self, pred_spec: torch.Tensor) -> torch.Tensor:
        """高性能优化处理主函数"""
        start_time = time.perf_counter()
        
        # 分割为块
        chunks = self.split_into_chunks(pred_spec)
        
        if len(chunks) == 1:
            # 单个块，直接使用CUDA Graph
            result = self.cuda_graph_vocoder.inference(chunks[0])
        else:
            # 多个块，批量处理
            audio_chunks = []
            chunk_start_time = time.perf_counter()
            
            for i, chunk in enumerate(chunks):
                audio_chunk = self.cuda_graph_vocoder.inference(chunk)
                audio_chunks.append(audio_chunk)
                
                # 每处理5个块输出一次进度
                if (i + 1) % 5 == 0:
                    progress = (i + 1) / len(chunks) * 100
                    print(f"🔄 处理进度: {progress:.1f}% ({i+1}/{len(chunks)})")
            
            chunk_end_time = time.perf_counter()
            chunk_duration = chunk_end_time - chunk_start_time
            
            # 记录性能历史
            self.performance_history.append(chunk_duration / len(chunks))
            if len(self.performance_history) > 10:
                self.performance_history.pop(0)
            
            # 合并块
            if self.overlap_ratio > 0:
                result = self.merge_overlapped_chunks(audio_chunks)
            else:
                result = torch.cat(audio_chunks, dim=-1)
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # 输出详细性能信息
        input_length = pred_spec.shape[-1]
        processing_speed = input_length / total_duration if total_duration > 0 else 0
        print(f"⚡ 优化处理完成: {total_duration:.3f}s, 速度: {processing_speed:.1f} frames/s, 块数: {len(chunks)}")
        
        return result

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)

# 检查是否启用并发模式
enable_concurrent = getattr(tts_config, 'enable_concurrent_tts', True)
num_gpus = getattr(tts_config, 'num_gpus', 4)

if False:  # 临时禁用并发模式 - 修复声音错误问题
    print(f"🚀 启用并发TTS模式，使用 {num_gpus} 个GPU")
    CONCURRENT_MODE = True
    # 初始化并发系统
    if not initialize_concurrent_tts(config_path, min(num_gpus, torch.cuda.device_count())):
        print("⚠️ 并发TTS初始化失败，回退到单线程模式")
        CONCURRENT_MODE = False
        tts_pipeline = TTS(tts_config)
    else:
        tts_pipeline = None  # 并发模式下不需要单一实例
else:
    print("📱 使用单线程TTS模式")
    CONCURRENT_MODE = False
    tts_pipeline = TTS(tts_config)

# 单线程模式的CUDA优化支持
if not CONCURRENT_MODE and tts_pipeline:
    tts_pipeline.use_vocoder_optimization = getattr(tts_config, 'use_vocoder_optimization', True)
    tts_pipeline.vocoder_chunk_size = getattr(tts_config, 'vocoder_chunk_size', 500)
    tts_pipeline.optimized_processor = None

def init_vocoder_optimization():
    """初始化高性能Vocoder优化"""
    if not tts_pipeline.use_vocoder_optimization or tts_pipeline.optimized_processor is not None:
        return
        
    if not CUDA_OPTIMIZATION_AVAILABLE:
        print("⚠️ CUDA优化不可用，跳过优化初始化")
        tts_pipeline.use_vocoder_optimization = False
        return
        
    try:
        # 获取vocoder模型
        vocoder_model = getattr(tts_pipeline, 'vocoder', None)
        if vocoder_model is None:
            print("⚠️ 无法获取vocoder模型，跳过优化")
            tts_pipeline.use_vocoder_optimization = False
            return
        
        # 获取配置参数
        chunk_size = getattr(tts_pipeline, 'vocoder_chunk_size', 500)
        overlap_ratio = getattr(tts_config, 'vocoder_overlap_ratio', 0.05)  # 5%重叠
        
        print(f"🔧 初始化优化配置: chunk_size={chunk_size}, overlap_ratio={overlap_ratio}")
            
        tts_pipeline.optimized_processor = OptimizedVocoderProcessor(
            vocoder_model,
            device=tts_config.device,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio
        )
        
        # 保存原始vocoder方法
        tts_pipeline._original_vocoder = tts_pipeline.vocoder
        
        # 创建优化的vocoder方法
        def optimized_vocoder_call(pred_spec):
            """高性能优化的vocoder调用"""
            if tts_pipeline.use_vocoder_optimization and tts_pipeline.optimized_processor is not None:
                try:
                    # 检查输入有效性
                    if pred_spec.numel() == 0:
                        print("⚠️ 输入频谱为空，回退到原始方法")
                        return tts_pipeline._original_vocoder(pred_spec)
                    
                    return tts_pipeline.optimized_processor.process_optimized(pred_spec)
                except Exception as e:
                    print(f"⚠️ 优化处理失败，回退到原始方法: {e}")
                    return tts_pipeline._original_vocoder(pred_spec)
            else:
                return tts_pipeline._original_vocoder(pred_spec)
        
        # 替换vocoder方法
        tts_pipeline.vocoder = optimized_vocoder_call
        
        print("✅ 高性能Vocoder优化初始化完成")
        
    except Exception as e:
        print(f"⚠️ Vocoder优化初始化失败: {e}")
        import traceback
        traceback.print_exc()
        tts_pipeline.use_vocoder_optimization = False

APP = FastAPI()

@APP.get("/performance_stats")
async def get_performance_stats():
    """获取性能统计信息"""
    stats = performance_stats.copy()
    if stats["total_requests"] > 0:
        stats["avg_tts_time"] = stats["total_tts_time"] / stats["total_requests"]
        stats["avg_t2s_time"] = stats["total_t2s_time"] / stats["total_requests"] 
        stats["avg_vocoder_time"] = stats["total_vocoder_time"] / stats["total_requests"]
        stats["avg_postprocess_time"] = stats["total_postprocess_time"] / stats["total_requests"]
        stats["requests_per_second"] = stats["total_requests"] / max(stats["total_tts_time"], 0.001)
    else:
        stats["avg_tts_time"] = 0
        stats["avg_t2s_time"] = 0
        stats["avg_vocoder_time"] = 0
        stats["avg_postprocess_time"] = 0
        stats["requests_per_second"] = 0
    
    # 添加优化状态
    stats["cuda_optimization_enabled"] = getattr(tts_pipeline, 'use_vocoder_optimization', False)
    stats["optimization_initialized"] = getattr(tts_pipeline, 'optimized_processor', None) is not None
    
    return stats

@APP.get("/concurrent_stats")
async def get_concurrent_stats():
    """获取并发处理统计信息"""
    if CONCURRENT_MODE:
        stats = get_concurrent_stats()
        stats["mode"] = "concurrent"
        stats["gpu_count"] = len(stats.get("gpu_stats", {}))
        return stats
    else:
        return {
            "mode": "single_thread",
            "gpu_count": 1,
            "message": "当前使用单线程模式"
        }

@APP.get("/optimization_status")
async def get_optimization_status():
    """获取详细的优化状态信息"""
    status = {
        "cuda_available": CUDA_OPTIMIZATION_AVAILABLE,
        "optimization_enabled": getattr(tts_pipeline, 'use_vocoder_optimization', False),
        "optimization_initialized": getattr(tts_pipeline, 'optimized_processor', None) is not None,
        "config": {
            "chunk_size": getattr(tts_pipeline, 'vocoder_chunk_size', 500),
            "overlap_ratio": getattr(tts_config, 'vocoder_overlap_ratio', 0.05)
        }
    }
    
    # 如果优化器已初始化，获取详细信息
    if status["optimization_initialized"]:
        processor = tts_pipeline.optimized_processor
        cuda_vocoder = processor.cuda_graph_vocoder
        
        status["optimization_details"] = {
            "cached_graph_shapes": len(cuda_vocoder.cached_graphs),
            "warmup_completed": cuda_vocoder.warmup_done,
            "performance_history_length": len(processor.performance_history),
            "avg_chunk_processing_time": sum(processor.performance_history) / len(processor.performance_history) if processor.performance_history else 0
        }
    
    return status

@APP.post("/toggle_optimization")
async def toggle_optimization(enable: bool = True):
    """动态启用/禁用CUDA优化"""
    try:
        if enable:
            tts_pipeline.use_vocoder_optimization = True
            init_vocoder_optimization()
            message = "CUDA优化已启用"
        else:
            tts_pipeline.use_vocoder_optimization = False
            if hasattr(tts_pipeline, '_original_vocoder'):
                tts_pipeline.vocoder = tts_pipeline._original_vocoder
            message = "CUDA优化已禁用"
        
        return {"success": True, "message": message}
    except Exception as e:
        return {"success": False, "message": f"操作失败: {e}"}


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = False  # v4模型不支持bucket处理，默认设为False
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler with concurrent processing support.

    Args:
        req (dict): TTS request parameters
    returns:
        StreamingResponse: audio stream response.
    """
    # 添加性能监控
    total_start_time = time.perf_counter()
    
    try:
        streaming_mode = req.get("streaming_mode", False)
        return_fragment = req.get("return_fragment", False)
        media_type = req.get("media_type", "wav")

        check_res = check_params(req)
        if check_res is not None:
            return check_res

        if streaming_mode or return_fragment:
            req["return_fragment"] = True

        # 并发模式处理
        if CONCURRENT_MODE:
            print("🚀 使用多GPU并发处理")
            return await _handle_concurrent_tts(req, total_start_time)
        
        # 单线程模式处理
        return await _handle_single_thread_tts(req, total_start_time)
        
    except Exception as e:
        print(f"❌ TTS处理失败: {e}")
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})

async def _handle_concurrent_tts(req: dict, total_start_time: float):
    """处理并发TTS请求"""
    try:
        # 使用并发处理
        response = await process_concurrent_tts(req)
        
        if response.success:
            # 更新性能统计
            total_duration = time.perf_counter() - total_start_time
            update_performance_stats("total", total_duration)
            
            print(f"🎯 并发处理完成: {total_duration:.3f}s (GPU {response.gpu_id}, 处理时间: {response.processing_time:.3f}s)")
            
            media_type = req.get("media_type", "wav")
            return Response(response.audio_data, media_type=f"audio/{media_type}")
        else:
            return JSONResponse(
                status_code=400, 
                content={"message": "concurrent tts failed", "Exception": response.error_message}
            )
            
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "concurrent tts failed", "Exception": str(e)})

async def _handle_single_thread_tts(req: dict, total_start_time: float):
    """处理单线程TTS请求"""
    try:
        # 初始化CUDA优化
        if tts_pipeline:
            init_vocoder_optimization()
        
        streaming_mode = req.get("streaming_mode", False)
        media_type = req.get("media_type", "wav")

        # 如果启用了优化，显示提示
        if hasattr(tts_pipeline, 'use_vocoder_optimization') and tts_pipeline.use_vocoder_optimization:
            print("🚀 使用单GPU CUDA优化处理")

        # T2S推理阶段
        t2s_start_time = time.perf_counter()
        tts_generator = tts_pipeline.run(req)
        t2s_end_time = time.perf_counter()
        t2s_duration = t2s_end_time - t2s_start_time
        update_performance_stats("t2s", t2s_duration)
        print(f"⏱️ T2S推理耗时: {t2s_duration:.3f}s")

        # Vocoder合成和音频后处理阶段
        vocoder_start_time = time.perf_counter()

        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                if_frist_chunk = True
                for sr, chunk in tts_generator:
                    if if_frist_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        media_type = "raw"
                        if_frist_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            vocoder_end_time = time.perf_counter()
            vocoder_duration = vocoder_end_time - vocoder_start_time
            update_performance_stats("vocoder", vocoder_duration)
            print(f"⏱️ Vocoder合成耗时: {vocoder_duration:.3f}s")

            # 总耗时
            total_end_time = time.perf_counter()
            total_duration = total_end_time - total_start_time
            update_performance_stats("total", total_duration)
            print(f"🎯 总耗时: {total_duration:.3f}s (T2S: {t2s_duration:.3f}s, Vocoder: {vocoder_duration:.3f}s)")

            return StreamingResponse(
                streaming_generator(tts_generator, media_type),
                media_type=f"audio/{media_type}",
            )
        else:
            # 非流式模式
            sr, audio_data = next(tts_generator)
            
            vocoder_end_time = time.perf_counter()
            vocoder_duration = vocoder_end_time - vocoder_start_time
            update_performance_stats("vocoder", vocoder_duration)
            print(f"⏱️ Vocoder合成耗时: {vocoder_duration:.3f}s")
            
            # 音频后处理阶段
            postprocess_start_time = time.perf_counter()
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            postprocess_end_time = time.perf_counter()
            postprocess_duration = postprocess_end_time - postprocess_start_time
            update_performance_stats("postprocess", postprocess_duration)
            print(f"⏱️ 音频后处理耗时: {postprocess_duration:.3f}s")
            
            # 总耗时
            total_end_time = time.perf_counter()
            total_duration = total_end_time - total_start_time
            update_performance_stats("total", total_duration)
            print(f"🎯 总耗时: {total_duration:.3f}s (T2S: {t2s_duration:.3f}s, Vocoder: {vocoder_duration:.3f}s, 后处理: {postprocess_duration:.3f}s)")
            
            return Response(audio_data, media_type=f"audio/{media_type}")
        
    except Exception as e:
        print(f"❌ 单线程TTS处理失败: {e}")
        return JSONResponse(status_code=400, content={"message": "single thread tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut0",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = False,  # v4模型不支持bucket处理
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.post("/upload_voice")
async def upload_voice(
    voice_name: str,
    audio_file: UploadFile = File(...),
    text_file: UploadFile = File(None),
    gender: str = "unknown",
    description: str = "",
    language: str = "zh"
):
    """上传新的音色"""
    try:
        # 检查文件类型
        if not audio_file.content_type.startswith("audio/"):
            return JSONResponse(status_code=400, content={"message": "Audio file type not supported"})
        
        # 创建voice目录
        voice_path = os.path.join("voice", voice_name)
        os.makedirs(voice_path, exist_ok=True)
        
        # 保存音频文件
        audio_filename = f"sample.{audio_file.filename.split('.')[-1]}"
        audio_path = os.path.join(voice_path, audio_filename)
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())
        
        # 保存文本文件
        text_filename = None
        if text_file:
            text_filename = "sample.wav.txt"
            text_path = os.path.join(voice_path, text_filename)
            with open(text_path, "wb") as f:
                f.write(await text_file.read())
        
        # 创建配置文件
        config = {
            "name": voice_name,
            "gender": gender,
            "description": description,
            "language": language
        }
        config_path = os.path.join(voice_path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 刷新缓存
        success, message = load_voice_to_cache(voice_name)
        
        return JSONResponse(status_code=200, content={
            "message": "success",
            "voice_name": voice_name,
            "audio_file": audio_filename,
            "text_file": text_filename,
            "cache_loaded": success
        })
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to upload voice", "Exception": str(e)})

@APP.delete("/voice/{voice_name}")
async def delete_voice(voice_name: str):
    """删除音色"""
    try:
        voice_path = os.path.join("voice", voice_name)
        if not os.path.exists(voice_path):
            return JSONResponse(status_code=404, content={"message": f"Voice '{voice_name}' not found"})
        
        # 从缓存中移除
        if voice_name in voice_cache:
            del voice_cache[voice_name]
        
        # 删除目录
        import shutil
        shutil.rmtree(voice_path)
        
        return JSONResponse(status_code=200, content={"message": f"Voice '{voice_name}' deleted successfully"})
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to delete voice", "Exception": str(e)})

@APP.post("/set_refer_audio")
async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
    try:
        # 检查文件类型，确保是音频文件
        if not audio_file.content_type.startswith("audio/"):
            return JSONResponse(status_code=400, content={"message": "file type is not supported"})

        os.makedirs("uploaded_audio", exist_ok=True)
        save_path = os.path.join("uploaded_audio", audio_file.filename)
        # 保存音频文件到服务器上的一个目录
        with open(save_path , "wb") as buffer:
            buffer.write(await audio_file.read())

        tts_pipeline.set_ref_audio(save_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# Voice Cache Management
voice_cache = {}  # 全局音色缓存

def load_voice_to_cache(voice_name: str):
    """将音色加载到内存缓存"""
    try:
        voice_info = get_voice_info(voice_name)
        if not voice_info:
            return False, f"Voice '{voice_name}' not found"
        
        voice_path = os.path.join("voice", voice_name)
        
        # 获取音频文件
        if not voice_info["audio_files"]:
            return False, f"No audio file found for voice '{voice_name}'"
        
        audio_file = voice_info["audio_files"][0]
        audio_path = os.path.join(voice_path, audio_file)
        
        if not os.path.exists(audio_path):
            return False, f"Audio file '{audio_path}' not found"
        
        # 读取音频文件到内存
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 读取文本文件
        possible_text_files = [
            audio_file.rsplit('.', 1)[0] + '.txt',  # sample.txt
            audio_file.rsplit('.', 1)[0] + '.txt',  # sample.wav.txt
            'sample.txt',  # 直接使用sample.txt
            'sample.wav.txt'  # 直接使用sample.wav.txt
        ]
        
        prompt_text = ""
        for text_file in possible_text_files:
            text_path = os.path.join(voice_path, text_file)
            if os.path.exists(text_path):
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        prompt_text = f.read().strip()
                    break
                except:
                    continue
        
        # 缓存音色信息
        voice_cache[voice_name] = {
            "audio_data": audio_data,
            "audio_path": audio_path,
            "prompt_text": prompt_text,
            "gender": voice_info.get("gender", "unknown"),
            "description": voice_info.get("description", ""),
            "language": voice_info.get("language", "zh"),
            "audio_filename": audio_file,
            "text_filename": text_file if prompt_text else None
        }
        
        return True, f"Voice '{voice_name}' loaded to cache successfully"
        
    except Exception as e:
        return False, f"Failed to load voice '{voice_name}': {str(e)}"

def refresh_voice_cache():
    """刷新所有音色缓存"""
    global voice_cache
    voice_cache.clear()
    
    voices = scan_voice_directory()
    success_count = 0
    error_count = 0
    errors = []
    
    for voice_name in voices.keys():
        success, message = load_voice_to_cache(voice_name)
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append(f"{voice_name}: {message}")
    
    return {
        "success_count": success_count,
        "error_count": error_count,
        "total_voices": len(voices),
        "errors": errors
    }

def get_cached_voice(voice_name: str):
    """获取缓存的音色信息"""
    return voice_cache.get(voice_name)

def set_cached_voice_reference(voice_name: str):
    """使用缓存的音色设置参考音频，如果缓存中没有则自动加载"""
    cached_voice = get_cached_voice(voice_name)
    if not cached_voice:
        # 缓存中没有，尝试从磁盘加载
        print(f"🔄 缓存中未找到音色 '{voice_name}'，正在从磁盘加载...")
        success, message = load_voice_to_cache(voice_name)
        if not success:
            raise ValueError(f"Voice '{voice_name}' not found in cache and failed to load from disk: {message}")
        
        # 重新获取缓存
        cached_voice = get_cached_voice(voice_name)
        if not cached_voice:
            raise ValueError(f"Voice '{voice_name}' failed to load into cache")
        
        print(f"✅ 音色 '{voice_name}' 已加载到缓存")
    
    # 🔧 修复：在并发模式下不需要设置全局pipeline，音频路径会传递给各个GPU工作器
    if not CONCURRENT_MODE and tts_pipeline:
        # 单线程模式才设置参考音频
        tts_pipeline.set_ref_audio(cached_voice["audio_path"])
        print(f"🎤 单线程模式设置参考音频: {cached_voice['audio_path']}")
    elif CONCURRENT_MODE:
        print(f"🎤 并发模式，音频路径将传递给GPU工作器: {cached_voice['audio_path']}")
    
    return cached_voice

# Voice Management Functions
def scan_voice_directory():
    """扫描voice目录，获取所有可用的voice"""
    voice_dir = "voice"
    voices = {}
    
    if not os.path.exists(voice_dir):
        return voices
    
    for voice_name in os.listdir(voice_dir):
        voice_path = os.path.join(voice_dir, voice_name)
        if os.path.isdir(voice_path):
            voice_info = {
                "name": voice_name,
                "audio_files": [],
                "text_files": [],
                "gender": "unknown"
            }
            
            # 扫描音频文件
            for file in os.listdir(voice_path):
                file_path = os.path.join(voice_path, file)
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    voice_info["audio_files"].append(file)
                elif file.lower().endswith('.txt'):
                    voice_info["text_files"].append(file)
            
            # 检查是否有配置文件
            config_file = os.path.join(voice_path, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        voice_info.update(config)
                except:
                    pass
            
            voices[voice_name] = voice_info
    
    return voices


def get_voice_info(voice_name: str):
    """获取指定voice的信息"""
    voices = scan_voice_directory()
    return voices.get(voice_name)


def set_voice_reference(voice_name: str, audio_file: str = None):
    """设置voice的参考音频"""
    voice_info = get_voice_info(voice_name)
    if not voice_info:
        raise ValueError(f"Voice '{voice_name}' not found")
    
    voice_path = os.path.join("voice", voice_name)
    
    # 如果没有指定音频文件，使用第一个音频文件
    if not audio_file and voice_info["audio_files"]:
        audio_file = voice_info["audio_files"][0]
    
    if not audio_file:
        raise ValueError(f"No audio file found for voice '{voice_name}'")
    
    audio_path = os.path.join(voice_path, audio_file)
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file '{audio_path}' not found")
    
    # 🔧 修复：在并发模式下不需要设置全局pipeline
    if not CONCURRENT_MODE and tts_pipeline:
        # 单线程模式才设置参考音频
        tts_pipeline.set_ref_audio(audio_path)
        print(f"🎤 单线程模式设置参考音频: {audio_path}")
    elif CONCURRENT_MODE:
        print(f"🎤 并发模式，音频路径将传递给GPU工作器: {audio_path}")
    
    # 如果有对应的文本文件，也设置prompt_text
    # 尝试多种可能的文本文件名
    possible_text_files = [
        audio_file.rsplit('.', 1)[0] + '.txt',  # sample.txt
        audio_file.rsplit('.', 1)[0] + '.txt',  # sample.wav.txt
        'sample.txt',  # 直接使用sample.txt
        'sample.wav.txt'  # 直接使用sample.wav.txt
    ]
    
    prompt_text = ""
    for text_file in possible_text_files:
        text_path = os.path.join(voice_path, text_file)
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
                print(f"Debug: 找到文本文件 {text_path}, 内容: {prompt_text}")
                break
            except Exception as e:
                print(f"Debug: 读取文本文件 {text_path} 失败: {e}")
                continue
    
    return {
        "audio_path": audio_path,
        "prompt_text": prompt_text,
        "gender": voice_info.get("gender", "unknown")
    }


# Voice Management Endpoints
@APP.get("/voices")
async def list_voices():
    """获取所有可用的voice列表"""
    try:
        voices = scan_voice_directory()
        return JSONResponse(status_code=200, content={"voices": voices})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to list voices", "Exception": str(e)})


@APP.get("/voice/{voice_name}")
async def get_voice(voice_name: str):
    """获取指定voice的详细信息"""
    try:
        voice_info = get_voice_info(voice_name)
        if not voice_info:
            return JSONResponse(status_code=404, content={"message": f"Voice '{voice_name}' not found"})
        return JSONResponse(status_code=200, content={"voice": voice_info})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to get voice", "Exception": str(e)})


@APP.post("/voice/{voice_name}/set")
async def set_voice(voice_name: str, audio_file: str = None):
    """设置指定的voice为当前参考音频"""
    try:
        voice_ref = set_voice_reference(voice_name, audio_file)
        return JSONResponse(status_code=200, content={
            "message": "success",
            "voice_name": voice_name,
            "audio_path": voice_ref["audio_path"],
            "prompt_text": voice_ref["prompt_text"],
            "gender": voice_ref["gender"]
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to set voice", "Exception": str(e)})


@APP.post("/refresh_voice_cache")
async def refresh_voice_cache_endpoint():
    """刷新音色缓存"""
    try:
        result = refresh_voice_cache()
        return JSONResponse(status_code=200, content={
            "message": "success",
            "result": result
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to refresh voice cache", "Exception": str(e)})

@APP.get("/performance_stats")
async def get_performance_stats():
    """获取性能统计信息"""
    try:
        avg_tts_time = performance_stats["total_tts_time"] / max(performance_stats["total_requests"], 1)
        avg_t2s_time = performance_stats["total_t2s_time"] / max(performance_stats["total_requests"], 1)
        avg_vocoder_time = performance_stats["total_vocoder_time"] / max(performance_stats["total_requests"], 1)
        avg_postprocess_time = performance_stats["total_postprocess_time"] / max(performance_stats["total_requests"], 1)
        
        return JSONResponse(status_code=200, content={
            "total_requests": performance_stats["total_requests"],
            "average_times": {
                "total_tts": round(avg_tts_time, 3),
                "t2s_inference": round(avg_t2s_time, 3),
                "vocoder_synthesis": round(avg_vocoder_time, 3),
                "audio_postprocess": round(avg_postprocess_time, 3)
            },
            "total_times": {
                "total_tts": round(performance_stats["total_tts_time"], 3),
                "t2s_inference": round(performance_stats["total_t2s_time"], 3),
                "vocoder_synthesis": round(performance_stats["total_vocoder_time"], 3),
                "audio_postprocess": round(performance_stats["total_postprocess_time"], 3)
            }
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to get performance stats", "Exception": str(e)})

@APP.post("/reset_performance_stats")
async def reset_performance_stats():
    """重置性能统计"""
    try:
        global performance_stats
        performance_stats = {
            "total_requests": 0,
            "total_tts_time": 0,
            "total_t2s_time": 0,
            "total_vocoder_time": 0,
            "total_postprocess_time": 0
        }
        return JSONResponse(status_code=200, content={"message": "Performance stats reset successfully"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to reset performance stats", "Exception": str(e)})

@APP.get("/voice_cache_status")
async def get_voice_cache_status():
    """获取音色缓存状态"""
    try:
        cached_voices = list(voice_cache.keys())
        return JSONResponse(status_code=200, content={
            "cached_voices": cached_voices,
            "cache_size": len(voice_cache),
            "cache_info": {name: {
                "audio_size": len(info["audio_data"]),
                "prompt_text": info["prompt_text"][:50] + "..." if len(info["prompt_text"]) > 50 else info["prompt_text"],
                "gender": info["gender"]
            } for name, info in voice_cache.items()}
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to get cache status", "Exception": str(e)})

@APP.post("/tts_with_cached_voice")
async def tts_with_cached_voice(
    voice_name: str,
    text: str,
    text_lang: str = "zh",
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = False,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False
):
    """使用缓存的音色进行TTS推理"""
    try:
        # 使用缓存的音色
        cached_voice = set_cached_voice_reference(voice_name)
        
        # 构建TTS请求
        req = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": cached_voice["audio_path"],
            "aux_ref_audio_paths": [],
            "prompt_text": cached_voice["prompt_text"],
            "prompt_lang": text_lang.lower(),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "batch_threshold": batch_threshold,
            "split_bucket": split_bucket,
            "speed_factor": speed_factor,
            "fragment_interval": fragment_interval,
            "seed": seed,
            "media_type": media_type,
            "streaming_mode": streaming_mode,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty,
            "sample_steps": sample_steps,
            "super_sampling": super_sampling,
        }
        
        return await tts_handle(req)
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "TTS with cached voice failed", "Exception": str(e)})

@APP.post("/tts_with_voice")
async def tts_with_voice(
    voice_name: str,
    text: str,
    text_lang: str = "zh",
    audio_file: str = None,
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = False,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False
):
    """使用指定voice进行TTS推理"""
    try:
        # 设置voice
        voice_ref = set_voice_reference(voice_name, audio_file)
        
        # 构建TTS请求
        req = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": voice_ref["audio_path"],
            "aux_ref_audio_paths": [],
            "prompt_text": voice_ref["prompt_text"],
            "prompt_lang": text_lang.lower(),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "batch_threshold": batch_threshold,
            "split_bucket": split_bucket,
            "speed_factor": speed_factor,
            "fragment_interval": fragment_interval,
            "seed": seed,
            "media_type": media_type,
            "streaming_mode": streaming_mode,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty,
            "sample_steps": sample_steps,
            "super_sampling": super_sampling,
        }
        
        return await tts_handle(req)
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "TTS with voice failed", "Exception": str(e)})


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
