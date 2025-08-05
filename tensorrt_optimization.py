#!/usr/bin/env python3
"""
TensorRT优化实现脚本
"""
import os
import sys
import time
import torch
import numpy as np
from typing import Optional, Dict, Any

# 添加项目路径
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

try:
    import torch_tensorrt
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️ TensorRT未安装，请先安装TensorRT")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

class TensorRTOptimizer:
    """TensorRT优化器"""
    
    def __init__(self, tts_pipeline: TTS):
        self.tts_pipeline = tts_pipeline
        self.trt_vocoder = None
        self.original_vocoder = None
        self.optimization_config = {
            "precision": "fp16",
            "workspace_size": 1 << 30,  # 1GB
            "max_batch_size": 4,
            "dynamic_shapes": True
        }
    
    def check_tensorrt_availability(self) -> bool:
        """检查TensorRT是否可用"""
        if not TENSORRT_AVAILABLE:
            print("❌ TensorRT未安装")
            return False
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用")
            return False
        
        print("✅ TensorRT环境检查通过")
        return True
    
    def get_vocoder_input_shape(self) -> tuple:
        """获取Vocoder输入形状"""
        version = self.tts_pipeline.configs.version
        
        if version == "v3":
            # BigVGAN输入形状
            return (1, 100, 468)  # (batch, channels, time)
        else:
            # Generator输入形状 (v4)
            return (1, 100, 500)  # (batch, channels, time)
    
    def get_dynamic_shapes(self) -> Dict[str, tuple]:
        """获取动态形状配置"""
        base_shape = self.get_vocoder_input_shape()
        batch_size, channels, time_steps = base_shape
        
        return {
            "min_shape": (1, channels, 100),      # 最小输入
            "opt_shape": (1, channels, time_steps), # 最优输入
            "max_shape": (1, channels, 2000)      # 最大输入
        }
    
    def convert_vocoder_to_tensorrt(self, precision: str = "fp16") -> bool:
        """转换Vocoder模型到TensorRT"""
        if not self.check_tensorrt_availability():
            return False
        
        try:
            print("🔄 开始转换Vocoder模型到TensorRT...")
            
            # 保存原始模型
            self.original_vocoder = self.tts_pipeline.vocoder
            
            # 获取输入形状
            input_shape = self.get_vocoder_input_shape()
            dynamic_shapes = self.get_dynamic_shapes()
            
            # 设置TensorRT配置
            if self.optimization_config["dynamic_shapes"]:
                trt_input = torch_tensorrt.Input(
                    min_shape=dynamic_shapes["min_shape"],
                    opt_shape=dynamic_shapes["opt_shape"],
                    max_shape=dynamic_shapes["max_shape"]
                )
            else:
                trt_input = torch_tensorrt.Input(input_shape)
            
            # 设置精度
            if precision == "fp16":
                enabled_precisions = [torch.float16]
            else:
                enabled_precisions = [torch.float32]
            
            # 转换模型
            self.trt_vocoder = torch_tensorrt.compile(
                self.original_vocoder,
                inputs=[trt_input],
                enabled_precisions=enabled_precisions,
                workspace_size=self.optimization_config["workspace_size"],
                max_batch_size=self.optimization_config["max_batch_size"]
            )
            
            print("✅ Vocoder模型转换成功!")
            return True
            
        except Exception as e:
            print(f"❌ 模型转换失败: {e}")
            return False
    
    def benchmark_performance(self, num_runs: int = 100) -> Dict[str, Any]:
        """性能基准测试"""
        if self.trt_vocoder is None:
            print("❌ TensorRT模型未初始化")
            return {}
        
        print(f"🧪 开始性能基准测试 ({num_runs} 次运行)...")
        
        # 准备测试数据
        input_shape = self.get_vocoder_input_shape()
        test_input = torch.randn(input_shape).cuda()
        
        # 测试原始模型
        print("📊 测试原始模型...")
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.original_vocoder(test_input)
        
        torch.cuda.synchronize()
        original_time = time.perf_counter() - start_time
        
        # 测试TensorRT模型
        print("📊 测试TensorRT模型...")
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.trt_vocoder(test_input)
        
        torch.cuda.synchronize()
        trt_time = time.perf_counter() - start_time
        
        # 计算性能指标
        speedup = original_time / trt_time
        throughput_original = num_runs / original_time
        throughput_trt = num_runs / trt_time
        
        results = {
            "original_time": original_time,
            "trt_time": trt_time,
            "speedup": speedup,
            "throughput_original": throughput_original,
            "throughput_trt": throughput_trt,
            "throughput_improvement": throughput_trt / throughput_original,
            "time_reduction_percent": (1 - trt_time / original_time) * 100
        }
        
        print(f"📈 性能测试结果:")
        print(f"   原始模型时间: {original_time:.3f}s")
        print(f"   TensorRT时间: {trt_time:.3f}s")
        print(f"   加速比: {speedup:.2f}x")
        print(f"   吞吐量提升: {results['throughput_improvement']:.2f}x")
        print(f"   时间减少: {results['time_reduction_percent']:.1f}%")
        
        return results
    
    def test_audio_quality(self, num_tests: int = 10) -> Dict[str, Any]:
        """音频质量测试"""
        if self.trt_vocoder is None:
            print("❌ TensorRT模型未初始化")
            return {}
        
        print(f"🎵 开始音频质量测试 ({num_tests} 次测试)...")
        
        # 准备测试数据
        input_shape = self.get_vocoder_input_shape()
        test_input = torch.randn(input_shape).cuda()
        
        quality_metrics = {
            "mse": [],
            "mae": [],
            "similarity": []
        }
        
        for i in range(num_tests):
            # 生成测试输入
            test_input = torch.randn(input_shape).cuda()
            
            # 生成音频
            with torch.no_grad():
                original_audio = self.original_vocoder(test_input)
                trt_audio = self.trt_vocoder(test_input)
            
            # 计算质量指标
            mse = torch.mean((original_audio - trt_audio) ** 2)
            mae = torch.mean(torch.abs(original_audio - trt_audio))
            similarity = 1 - mae.item()
            
            quality_metrics["mse"].append(mse.item())
            quality_metrics["mae"].append(mae.item())
            quality_metrics["similarity"].append(similarity)
        
        # 计算平均指标
        avg_metrics = {
            "avg_mse": np.mean(quality_metrics["mse"]),
            "avg_mae": np.mean(quality_metrics["mae"]),
            "avg_similarity": np.mean(quality_metrics["similarity"]),
            "std_mse": np.std(quality_metrics["mse"]),
            "std_mae": np.std(quality_metrics["mae"]),
            "std_similarity": np.std(quality_metrics["similarity"])
        }
        
        print(f"🎵 音频质量测试结果:")
        print(f"   平均MSE: {avg_metrics['avg_mse']:.6f} ± {avg_metrics['std_mse']:.6f}")
        print(f"   平均MAE: {avg_metrics['avg_mae']:.6f} ± {avg_metrics['std_mae']:.6f}")
        print(f"   平均相似度: {avg_metrics['avg_similarity']:.4f} ± {avg_metrics['std_similarity']:.4f}")
        
        return avg_metrics
    
    def save_optimized_model(self, save_path: str) -> bool:
        """保存优化后的模型"""
        if self.trt_vocoder is None:
            print("❌ TensorRT模型未初始化")
            return False
        
        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存模型
            torch.save(self.trt_vocoder.state_dict(), save_path)
            print(f"✅ TensorRT模型已保存到: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
            return False
    
    def load_optimized_model(self, load_path: str) -> bool:
        """加载优化后的模型"""
        if not os.path.exists(load_path):
            print(f"❌ 模型文件不存在: {load_path}")
            return False
        
        try:
            # 加载模型
            self.trt_vocoder.load_state_dict(torch.load(load_path))
            print(f"✅ TensorRT模型已加载: {load_path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def integrate_with_tts(self) -> bool:
        """集成到TTS管道"""
        if self.trt_vocoder is None:
            print("❌ TensorRT模型未初始化")
            return False
        
        try:
            # 替换原始vocoder
            self.tts_pipeline.vocoder = self.trt_vocoder
            print("✅ TensorRT模型已集成到TTS管道")
            return True
            
        except Exception as e:
            print(f"❌ 集成失败: {e}")
            return False
    
    def restore_original_model(self) -> bool:
        """恢复原始模型"""
        if self.original_vocoder is None:
            print("❌ 原始模型未保存")
            return False
        
        try:
            self.tts_pipeline.vocoder = self.original_vocoder
            print("✅ 已恢复原始模型")
            return True
            
        except Exception as e:
            print(f"❌ 恢复失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 TensorRT优化工具")
    print("=" * 50)
    
    # 检查TensorRT可用性
    if not TENSORRT_AVAILABLE:
        print("❌ TensorRT未安装，请先安装TensorRT")
        print("安装命令:")
        print("  pip install tensorrt")
        print("  或")
        print("  conda install -c conda-forge tensorrt")
        return
    
    # 初始化TTS管道
    print("📦 初始化TTS管道...")
    try:
        config_path = "GPT_SoVITS/configs/tts_infer.yaml"
        tts_config = TTS_Config(config_path)
        tts_pipeline = TTS(tts_config)
        print(f"✅ TTS管道初始化成功 (版本: {tts_config.version})")
    except Exception as e:
        print(f"❌ TTS管道初始化失败: {e}")
        return
    
    # 创建优化器
    optimizer = TensorRTOptimizer(tts_pipeline)
    
    # 转换模型
    print("\n🔄 开始模型转换...")
    if not optimizer.convert_vocoder_to_tensorrt():
        print("❌ 模型转换失败")
        return
    
    # 性能测试
    print("\n🧪 开始性能测试...")
    performance_results = optimizer.benchmark_performance(num_runs=50)
    
    # 质量测试
    print("\n🎵 开始质量测试...")
    quality_results = optimizer.test_audio_quality(num_tests=5)
    
    # 保存模型
    print("\n💾 保存优化模型...")
    save_path = "optimized_models/trt_vocoder.pth"
    optimizer.save_optimized_model(save_path)
    
    # 生成报告
    print("\n📊 优化报告")
    print("=" * 50)
    print(f"模型版本: {tts_config.version}")
    print(f"优化精度: {optimizer.optimization_config['precision']}")
    print(f"工作空间大小: {optimizer.optimization_config['workspace_size'] / (1<<30):.1f}GB")
    
    if performance_results:
        print(f"\n性能提升:")
        print(f"  加速比: {performance_results['speedup']:.2f}x")
        print(f"  吞吐量提升: {performance_results['throughput_improvement']:.2f}x")
        print(f"  时间减少: {performance_results['time_reduction_percent']:.1f}%")
    
    if quality_results:
        print(f"\n质量指标:")
        print(f"  平均相似度: {quality_results['avg_similarity']:.4f}")
        print(f"  平均MSE: {quality_results['avg_mse']:.6f}")
    
    print("\n✅ 优化完成!")

if __name__ == "__main__":
    main() 