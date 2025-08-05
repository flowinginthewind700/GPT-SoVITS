#!/usr/bin/env python3
"""
并发TTS增强模块 - 支持多GPU并发处理
"""
import asyncio
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import torch
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class TTSRequest:
    """TTS请求数据结构"""
    request_id: str
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    temperature: float
    speed_factor: float
    sample_steps: int
    media_type: str
    other_params: dict

@dataclass
class TTSResponse:
    """TTS响应数据结构"""
    request_id: str
    success: bool
    audio_data: Optional[bytes]
    error_message: Optional[str]
    processing_time: float
    gpu_id: int

class GPUWorker:
    """单GPU工作进程"""
    
    def __init__(self, gpu_id: int, config_path: str):
        self.gpu_id = gpu_id
        self.config_path = config_path
        self.device = f"cuda:{gpu_id}"
        self.tts_pipeline = None
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
    def initialize(self):
        """初始化GPU工作器"""
        try:
            # 设置当前GPU
            torch.cuda.set_device(self.gpu_id)
            
            # 导入必要模块（避免在主进程导入）
            import sys
            import os
            sys.path.append(os.getcwd())
            sys.path.append("GPT_SoVITS")
            
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            
            # 创建配置
            config = TTS_Config(self.config_path)
            config.device = self.device
            
            # 创建TTS实例
            self.tts_pipeline = TTS(config)
            
            # 初始化CUDA优化（如果可用）
            self._init_gpu_optimization()
            
            print(f"✅ GPU {self.gpu_id} 工作器初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ GPU {self.gpu_id} 工作器初始化失败: {e}")
            return False
    
    def _init_gpu_optimization(self):
        """初始化GPU特定的优化"""
        try:
            # 设置CUDA优化
            if hasattr(self.tts_pipeline, 'use_vocoder_optimization'):
                self.tts_pipeline.use_vocoder_optimization = True
                
            # 预热GPU
            dummy_req = {
                "text": "预热测试",
                "text_lang": "zh",
                "ref_audio_path": "voice/vivienne/sample.mp3",
                "prompt_text": "Hello",
                "prompt_lang": "en",
                "temperature": 1.0,
                "speed_factor": 1.0,
                "sample_steps": 32
            }
            
            # 执行预热
            next(self.tts_pipeline.run(dummy_req))
            print(f"🔥 GPU {self.gpu_id} 预热完成")
            
        except Exception as e:
            print(f"⚠️ GPU {self.gpu_id} 优化初始化失败: {e}")
    
    def start(self):
        """启动工作器"""
        if not self.initialize():
            return False
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        return True
    
    def stop(self):
        """停止工作器"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _worker_loop(self):
        """工作器主循环"""
        while self.is_running:
            try:
                # 获取请求（超时机制）
                request = self.request_queue.get(timeout=1.0)
                
                # 处理请求
                response = self._process_request(request)
                
                # 返回响应
                self.response_queue.put(response)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ GPU {self.gpu_id} 处理请求失败: {e}")
    
    def _process_request(self, request: TTSRequest) -> TTSResponse:
        """处理单个TTS请求"""
        start_time = time.perf_counter()
        
        try:
            # 🔧 关键修复：每次请求前设置参考音频
            if request.ref_audio_path:
                self.tts_pipeline.set_ref_audio(request.ref_audio_path)
                print(f"🎤 GPU {self.gpu_id} 设置参考音频: {request.ref_audio_path}")
            
            # 构建TTS参数
            tts_params = {
                "text": request.text,
                "text_lang": request.text_lang,
                "ref_audio_path": request.ref_audio_path,
                "prompt_text": request.prompt_text,
                "prompt_lang": request.prompt_lang,
                "temperature": request.temperature,
                "speed_factor": request.speed_factor,
                "sample_steps": request.sample_steps,
                **request.other_params
            }
            
            # 执行TTS
            sr, audio_data = next(self.tts_pipeline.run(tts_params))
            
            # 转换为字节数据
            import numpy as np
            from io import BytesIO
            import soundfile as sf
            
            audio_buffer = BytesIO()
            sf.write(audio_buffer, audio_data, sr, format=request.media_type)
            audio_bytes = audio_buffer.getvalue()
            
            processing_time = time.perf_counter() - start_time
            
            return TTSResponse(
                request_id=request.request_id,
                success=True,
                audio_data=audio_bytes,
                error_message=None,
                processing_time=processing_time,
                gpu_id=self.gpu_id
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return TTSResponse(
                request_id=request.request_id,
                success=False,
                audio_data=None,
                error_message=str(e),
                processing_time=processing_time,
                gpu_id=self.gpu_id
            )
    
    def submit_request(self, request: TTSRequest) -> bool:
        """提交请求到队列"""
        try:
            self.request_queue.put_nowait(request)
            return True
        except queue.Full:
            return False
    
    def get_response(self) -> Optional[TTSResponse]:
        """获取处理结果"""
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_queue_size(self) -> Tuple[int, int]:
        """获取队列大小"""
        return self.request_queue.qsize(), self.response_queue.qsize()

class ConcurrentTTSManager:
    """并发TTS管理器"""
    
    def __init__(self, config_path: str, num_gpus: int = 4):
        self.config_path = config_path
        self.num_gpus = num_gpus
        self.workers: List[GPUWorker] = []
        self.response_futures: Dict[str, asyncio.Future] = {}
        self.load_balancer_index = 0
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "gpu_stats": {}
        }
        
    def initialize(self) -> bool:
        """初始化所有GPU工作器"""
        print(f"🚀 初始化 {self.num_gpus} 个GPU工作器...")
        
        for gpu_id in range(self.num_gpus):
            worker = GPUWorker(gpu_id, self.config_path)
            if worker.start():
                self.workers.append(worker)
                self.stats["gpu_stats"][gpu_id] = {
                    "requests_processed": 0,
                    "total_processing_time": 0,
                    "average_processing_time": 0
                }
                print(f"✅ GPU {gpu_id} 工作器启动成功")
            else:
                print(f"❌ GPU {gpu_id} 工作器启动失败")
        
        if not self.workers:
            print("❌ 没有可用的GPU工作器")
            return False
            
        # 启动响应处理循环
        asyncio.create_task(self._response_handler_loop())
        
        print(f"🎉 并发TTS管理器初始化完成，可用GPU: {len(self.workers)}")
        return True
    
    def shutdown(self):
        """关闭所有工作器"""
        print("🔄 关闭所有GPU工作器...")
        for worker in self.workers:
            worker.stop()
        self.workers.clear()
    
    async def process_request(self, request: TTSRequest) -> TTSResponse:
        """处理TTS请求（异步）"""
        # 选择工作器
        worker = self._select_worker()
        if not worker:
            return TTSResponse(
                request_id=request.request_id,
                success=False,
                audio_data=None,
                error_message="没有可用的GPU工作器",
                processing_time=0,
                gpu_id=-1
            )
        
        # 提交请求
        if not worker.submit_request(request):
            return TTSResponse(
                request_id=request.request_id,
                success=False,
                audio_data=None,
                error_message="工作器队列已满",
                processing_time=0,
                gpu_id=worker.gpu_id
            )
        
        # 创建Future等待响应
        future = asyncio.Future()
        self.response_futures[request.request_id] = future
        self.stats["total_requests"] += 1
        
        try:
            # 等待处理完成
            response = await future
            
            # 更新统计
            if response.success:
                self.stats["completed_requests"] += 1
                gpu_stats = self.stats["gpu_stats"][response.gpu_id]
                gpu_stats["requests_processed"] += 1
                gpu_stats["total_processing_time"] += response.processing_time
                gpu_stats["average_processing_time"] = (
                    gpu_stats["total_processing_time"] / gpu_stats["requests_processed"]
                )
            else:
                self.stats["failed_requests"] += 1
            
            return response
            
        finally:
            # 清理Future
            if request.request_id in self.response_futures:
                del self.response_futures[request.request_id]
    
    def _select_worker(self) -> Optional[GPUWorker]:
        """选择最优工作器（负载均衡）"""
        if not self.workers:
            return None
        
        # 简单的轮询负载均衡
        worker = self.workers[self.load_balancer_index]
        self.load_balancer_index = (self.load_balancer_index + 1) % len(self.workers)
        
        # 检查队列大小，如果太满则尝试下一个
        req_queue_size, _ = worker.get_queue_size()
        if req_queue_size > 10:  # 队列太满，尝试找其他GPU
            for _ in range(len(self.workers)):
                worker = self.workers[self.load_balancer_index]
                self.load_balancer_index = (self.load_balancer_index + 1) % len(self.workers)
                req_queue_size, _ = worker.get_queue_size()
                if req_queue_size <= 10:
                    break
        
        return worker
    
    async def _response_handler_loop(self):
        """响应处理循环"""
        while True:
            try:
                # 检查所有工作器的响应
                for worker in self.workers:
                    response = worker.get_response()
                    if response:
                        # 找到对应的Future并设置结果
                        future = self.response_futures.get(response.request_id)
                        if future and not future.done():
                            future.set_result(response)
                
                # 短暂休眠避免CPU占用过高
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"❌ 响应处理循环错误: {e}")
                await asyncio.sleep(0.1)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加实时队列状态
        stats["queue_status"] = {}
        for i, worker in enumerate(self.workers):
            req_size, resp_size = worker.get_queue_size()
            stats["queue_status"][f"gpu_{i}"] = {
                "request_queue": req_size,
                "response_queue": resp_size
            }
        
        # 计算总体性能
        if stats["completed_requests"] > 0:
            total_time = sum(gpu["total_processing_time"] for gpu in stats["gpu_stats"].values())
            stats["overall_average_time"] = total_time / stats["completed_requests"]
            stats["success_rate"] = stats["completed_requests"] / stats["total_requests"] * 100
        else:
            stats["overall_average_time"] = 0
            stats["success_rate"] = 0
        
        return stats

# 全局并发管理器实例
concurrent_manager: Optional[ConcurrentTTSManager] = None

def initialize_concurrent_tts(config_path: str, num_gpus: int = 4) -> bool:
    """初始化并发TTS系统"""
    global concurrent_manager
    
    try:
        concurrent_manager = ConcurrentTTSManager(config_path, num_gpus)
        return concurrent_manager.initialize()
    except Exception as e:
        print(f"❌ 并发TTS系统初始化失败: {e}")
        return False

def shutdown_concurrent_tts():
    """关闭并发TTS系统"""
    global concurrent_manager
    if concurrent_manager:
        concurrent_manager.shutdown()
        concurrent_manager = None

async def process_concurrent_tts(request_data: dict) -> TTSResponse:
    """处理并发TTS请求"""
    if not concurrent_manager:
        raise RuntimeError("并发TTS系统未初始化")
    
    # 创建请求对象
    request = TTSRequest(
        request_id=f"req_{int(time.time() * 1000000)}",
        text=request_data.get("text", ""),
        text_lang=request_data.get("text_lang", "auto"),
        ref_audio_path=request_data.get("ref_audio_path", ""),
        prompt_text=request_data.get("prompt_text", ""),
        prompt_lang=request_data.get("prompt_lang", "zh"),
        temperature=request_data.get("temperature", 1.0),
        speed_factor=request_data.get("speed_factor", 1.0),
        sample_steps=request_data.get("sample_steps", 32),
        media_type=request_data.get("media_type", "wav"),
        other_params={
            k: v for k, v in request_data.items() 
            if k not in ["text", "text_lang", "ref_audio_path", "prompt_text", 
                        "prompt_lang", "temperature", "speed_factor", "sample_steps", "media_type"]
        }
    )
    
    # 处理请求
    return await concurrent_manager.process_request(request)

def get_concurrent_stats() -> dict:
    """获取并发统计信息"""
    if not concurrent_manager:
        return {"error": "并发TTS系统未初始化"}
    
    return concurrent_manager.get_stats()

if __name__ == "__main__":
    print("🚀 并发TTS增强模块")
    print("主要功能:")
    print("1. 多GPU并发处理")
    print("2. 智能负载均衡") 
    print("3. 异步请求处理")
    print("4. 实时性能监控")
    print("\n预期性能提升: 4x (4个GPU并发)")