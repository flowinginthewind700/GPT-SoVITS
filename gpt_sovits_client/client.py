"""
GPT-SoVITS Client SDK - Main Client
"""

import os
import time
import requests
import re
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin

from .models import TTSRequest, TTSResponse, LanguageType, TextSplitMethod
from .exceptions import (
    GPTSoVITSException, APIException, ValidationException, 
    ConnectionException, TimeoutException, FileNotFoundException
)


class GPTSoVITSClient:
    """GPT-SoVITS客户端"""
    
    def __init__(self, base_url: str = "http://localhost:9880", timeout: int = 180):
        """
        初始化客户端
        
        Args:
            base_url: API服务地址
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """发送HTTP请求"""
        url = urljoin(self.base_url, endpoint)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            return response
        except requests.exceptions.Timeout:
            raise TimeoutException(f"请求超时: {url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionException(f"连接失败: {url}")
        except Exception as e:
            raise GPTSoVITSException(f"请求异常: {str(e)}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = self._make_request("GET", "/docs")
            return response.status_code == 200
        except:
            return False
    
    def detect_language_segments(self, text: str) -> List[Dict[str, str]]:
        """
        检测文本中的语言片段
        
        Args:
            text: 要检测的文本
            
        Returns:
            语言片段列表: [{"text": "Hello", "lang": "en"}, {"text": "你好", "lang": "zh"}]
        """
        segments = []
        
        # 语言检测规则
        japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF]+'  # 平假名、片假名
        korean_pattern = r'[\uAC00-\uD7AF]+'  # 韩文
        chinese_pattern = r'[\u4E00-\u9FAF]+'  # 汉字
        english_pattern = r'[a-zA-Z]+'  # 英文
        
        # 分割文本
        current_pos = 0
        while current_pos < len(text):
            # 跳过空白字符
            while current_pos < len(text) and text[current_pos].isspace():
                current_pos += 1
            
            if current_pos >= len(text):
                break
            
            # 检测当前字符的语言
            char = text[current_pos]
            lang = None
            
            if re.match(japanese_pattern, char):
                # 日语（平假名或片假名）
                jp_match = re.match(r'[\u3040-\u309F\u30A0-\u30FF]+', text[current_pos:])
                if jp_match:
                    lang = "ja"
                    segment_text = jp_match.group()
            elif re.match(korean_pattern, char):
                # 韩文
                ko_match = re.match(r'[\uAC00-\uD7AF]+', text[current_pos:])
                if ko_match:
                    lang = "ko"
                    segment_text = ko_match.group()
            elif re.match(chinese_pattern, char):
                # 中文（不包含日韩字符）
                zh_match = re.match(r'[\u4E00-\u9FAF]+', text[current_pos:])
                if zh_match:
                    lang = "zh"
                    segment_text = zh_match.group()
            elif re.match(english_pattern, char):
                # 英文
                en_match = re.match(r'[a-zA-Z\s]+', text[current_pos:])
                if en_match:
                    lang = "en"
                    segment_text = en_match.group().strip()
            else:
                # 其他字符（标点符号等）
                lang = "other"
                segment_text = char
            
            if lang and lang != "other":
                segments.append({
                    "text": segment_text,
                    "lang": lang
                })
            
            current_pos += len(segment_text)
        
        return segments
    
    def is_mixed_language(self, text: str) -> bool:
        """检查是否为混合语言文本"""
        segments = self.detect_language_segments(text)
        languages = set(seg["lang"] for seg in segments)
        return len(languages) > 1
    
    def get_primary_language(self, text: str) -> str:
        """获取主要语言"""
        segments = self.detect_language_segments(text)
        if not segments:
            return "zh"
        
        # 统计各语言字符数
        lang_counts = {}
        for seg in segments:
            lang = seg["lang"]
            count = len(seg["text"])
            lang_counts[lang] = lang_counts.get(lang, 0) + count
        
        # 返回字符数最多的语言
        return max(lang_counts.items(), key=lambda x: x[1])[0]
    
    def auto_detect_language(self, text: str) -> LanguageType:
        """自动检测语言类型"""
        if self.is_mixed_language(text):
            return LanguageType.AUTO
        
        primary_lang = self.get_primary_language(text)
        if primary_lang == "zh":
            return LanguageType.ALL_ZH
        elif primary_lang == "en":
            return LanguageType.EN
        elif primary_lang == "ja":
            return LanguageType.JA
        elif primary_lang == "ko":
            return LanguageType.KO
        else:
            return LanguageType.AUTO
    
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        执行TTS合成
        
        Args:
            request: TTS请求参数
            
        Returns:
            TTS响应结果
        """
        start_time = time.time()
        
        try:
            # 验证请求参数
            request.validate()
            
            # 自动检测语言（如果设置为AUTO）
            if request.text_lang == LanguageType.AUTO:
                request.text_lang = self.auto_detect_language(request.text)
            
            if request.prompt_lang == LanguageType.AUTO:
                if request.prompt_text:
                    request.prompt_lang = self.auto_detect_language(request.prompt_text)
                else:
                    request.prompt_lang = request.text_lang
            
            # 发送请求
            payload = request.to_dict()
            response = self._make_request("POST", "/tts", json=payload)
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                # 保存音频文件
                output_path = f"tts_output_{int(time.time())}.wav"
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                
                return TTSResponse.success_response(
                    audio_path=output_path,
                    file_size=file_size,
                    message="TTS合成成功"
                )
            else:
                error_msg = response.text
                try:
                    error_json = response.json()
                    error_msg = error_json.get('message', str(error_json))
                except:
                    pass
                
                return TTSResponse.error_response(
                    message=f"API错误: {response.status_code} - {error_msg}",
                    error_code=response.status_code
                )
                
        except ValidationException as e:
            return TTSResponse.error_response(f"参数验证失败: {str(e)}")
        except FileNotFoundException as e:
            return TTSResponse.error_response(f"文件不存在: {str(e)}")
        except TimeoutException as e:
            return TTSResponse.error_response(f"请求超时: {str(e)}")
        except ConnectionException as e:
            return TTSResponse.error_response(f"连接失败: {str(e)}")
        except Exception as e:
            return TTSResponse.error_response(f"未知错误: {str(e)}")
    
    def synthesize_text(self, 
                       text: str, 
                       ref_audio_path: str,
                       text_lang: Union[LanguageType, str] = LanguageType.AUTO,
                       prompt_text: str = "",
                       prompt_lang: Union[LanguageType, str] = LanguageType.AUTO,
                       **kwargs) -> TTSResponse:
        """
        简化版TTS合成
        
        Args:
            text: 要合成的文本
            ref_audio_path: 参考音频路径
            text_lang: 文本语言
            prompt_text: 提示文本
            prompt_lang: 提示语言
            **kwargs: 其他参数
            
        Returns:
            TTS响应结果
        """
        # 转换语言类型
        if isinstance(text_lang, str):
            text_lang = LanguageType(text_lang)
        if isinstance(prompt_lang, str):
            prompt_lang = LanguageType(prompt_lang)
        
        # 创建请求对象
        request = TTSRequest(
            text=text,
            ref_audio_path=ref_audio_path,
            text_lang=text_lang,
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            **kwargs
        )
        
        return self.synthesize(request)
    
    def batch_synthesize(self, requests: List[TTSRequest]) -> List[TTSResponse]:
        """
        批量TTS合成
        
        Args:
            requests: TTS请求列表
            
        Returns:
            TTS响应列表
        """
        results = []
        for request in requests:
            result = self.synthesize(request)
            results.append(result)
        return results 