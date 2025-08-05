"""
GPT-SoVITS Client SDK - Data Models
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
import os


class LanguageType(str, Enum):
    """支持的语言类型"""
    AUTO = "auto"
    ALL_ZH = "all_zh"
    EN = "en"
    JA = "ja"
    KO = "ko"
    YUE = "yue"
    ALL_JA = "all_ja"
    ALL_KO = "all_ko"
    ALL_YUE = "all_yue"


class TextSplitMethod(str, Enum):
    """文本分割方法"""
    CUT0 = "cut0"  # 按标点符号分割
    CUT1 = "cut1"  # 按句子分割
    CUT2 = "cut2"  # 按字符数分割
    CUT3 = "cut3"  # 按语义分割
    CUT4 = "cut4"  # 按段落分割
    CUT5 = "cut5"  # 智能分割（推荐）


@dataclass
class TTSRequest:
    """TTS请求参数"""
    text: str
    ref_audio_path: str
    text_lang: LanguageType = LanguageType.AUTO
    prompt_text: str = ""
    prompt_lang: LanguageType = LanguageType.AUTO
    text_split_method: TextSplitMethod = TextSplitMethod.CUT5
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    batch_size: int = 1
    speed_factor: float = 1.0
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "text": self.text,
            "text_lang": self.text_lang.value,
            "ref_audio_path": os.path.abspath(self.ref_audio_path),
            "prompt_lang": self.prompt_lang.value,
            "prompt_text": self.prompt_text,
            "text_split_method": self.text_split_method.value,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "speed_factor": self.speed_factor,
            "seed": self.seed,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
            "parallel_infer": self.parallel_infer,
            "repetition_penalty": self.repetition_penalty,
            "sample_steps": self.sample_steps,
            "super_sampling": self.super_sampling
        }
    
    def validate(self) -> None:
        """验证请求参数"""
        if not self.text or not self.text.strip():
            raise ValueError("文本不能为空")
        
        if not os.path.exists(self.ref_audio_path):
            raise FileNotFoundError(f"参考音频文件不存在: {self.ref_audio_path}")
        
        if self.top_k < 1:
            raise ValueError("top_k必须大于0")
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p必须在0-1之间")
        
        if self.temperature <= 0:
            raise ValueError("temperature必须大于0")
        
        if self.batch_size < 1:
            raise ValueError("batch_size必须大于0")
        
        if self.speed_factor <= 0:
            raise ValueError("speed_factor必须大于0")


@dataclass
class TTSResponse:
    """TTS响应结果"""
    success: bool
    audio_path: Optional[str] = None
    audio_data: Optional[bytes] = None
    file_size: Optional[int] = None
    duration: Optional[float] = None
    message: str = ""
    error_code: Optional[int] = None
    processing_time: Optional[float] = None
    
    @classmethod
    def success_response(cls, audio_path: str, file_size: int, message: str = "生成成功") -> "TTSResponse":
        """创建成功响应"""
        return cls(
            success=True,
            audio_path=audio_path,
            file_size=file_size,
            message=message
        )
    
    @classmethod
    def error_response(cls, message: str, error_code: Optional[int] = None) -> "TTSResponse":
        """创建错误响应"""
        return cls(
            success=False,
            message=message,
            error_code=error_code
        ) 