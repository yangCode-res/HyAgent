# INSERT_YOUR_CODE

import logging
import os

# 设置日志文件路径（全局日志存储于 logs/hygraph.log）
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "hygraph.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# 创建 logger 并做全局一次性配置
_logger = logging.getLogger("hygraph-global-logger")
_logger.setLevel(logging.INFO)

# 防止多次添加 handler
if not _logger.handlers:
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)
    _logger.propagate = False
# ---- 降噪第三方库日志 ----
# 这些库默认会在 INFO 打一堆无用信息，比如 HTTP 请求、线程数之类
_noisy_loggers = [
    "httpx",
    "httpcore",        # httpx 底层也会打
    "numexpr",
    "numexpr.utils",
]

for name in _noisy_loggers:
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)   # 只保留 WARNING 及以上
    lg.propagate = False           # 不再往 root logger 冒泡
def get_global_logger():
    """
    获取全局 logger。
    用法示例：
        from hygraph.Logger.index import get_global_logger
        logger = get_global_logger()
        logger.info("Hello, logger!")
    """
    return _logger

