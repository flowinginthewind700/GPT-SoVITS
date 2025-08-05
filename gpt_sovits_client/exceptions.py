"""
GPT-SoVITS Client SDK - Exceptions
"""


class GPTSoVITSException(Exception):
    """GPT-SoVITS客户端基础异常"""
    pass


class APIException(GPTSoVITSException):
    """API调用异常"""
    def __init__(self, message: str, status_code: int = None, response_text: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class ValidationException(GPTSoVITSException):
    """参数验证异常"""
    pass


class ConnectionException(GPTSoVITSException):
    """连接异常"""
    pass


class TimeoutException(GPTSoVITSException):
    """超时异常"""
    pass


class FileNotFoundException(GPTSoVITSException):
    """文件不存在异常"""
    pass 