"""
业务异常定义

类似 Java 项目里的 BusinessException。
"""


class BusinessException(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class SecurityException(Exception):
    """认证/授权异常，必须保留真实 HTTP 401/403 语义。"""

    def __init__(self, *, status_code: int, code: int, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class AuthenticationException(SecurityException):
    def __init__(self, message: str = "未提供有效的访问凭据"):
        super().__init__(status_code=401, code=40101, message=message)


class AuthorizationException(SecurityException):
    def __init__(self, message: str = "当前身份没有执行该操作的权限"):
        super().__init__(status_code=403, code=40301, message=message)


ERROR_TYPE_MODEL_CALL_FAILED = "MODEL_CALL_FAILED"
ERROR_TYPE_STRUCTURED_JSON_INVALID = "STRUCTURED_JSON_INVALID"
ERROR_TYPE_STRUCTURED_FIELD_INVALID = "STRUCTURED_FIELD_INVALID"
ERROR_TYPE_TASK_TIMEOUT = "TASK_TIMEOUT"
ERROR_TYPE_WORKER_EXECUTION_ERROR = "WORKER_EXECUTION_ERROR"


class ModelCallException(BusinessException):
    def __init__(
        self,
        message: str = "模型调用失败",
        error_type: str = ERROR_TYPE_MODEL_CALL_FAILED,
    ):
        self.error_type = error_type
        super().__init__(code=50001, message=message)
