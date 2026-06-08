"""
业务异常定义

类似 Java 项目里的 BusinessException。
"""


class BusinessException(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ModelCallException(BusinessException):
    def __init__(self, message: str = "模型调用失败"):
        super().__init__(code=50001, message=message)
