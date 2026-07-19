"""
雪花 ID 生成器

用于生成分布式环境下趋势递增的业务 ID。
"""

import threading
import time


class SnowflakeIdGenerator:
    """
    简化版雪花 ID。

    结构：
    时间戳部分 + 机器 ID 部分 + 毫秒内序列号部分
    """

    def __init__(self, worker_id: int = 1):
        if worker_id < 0 or worker_id > 1023:
            raise ValueError("worker_id 必须在 0 到 1023 之间")

        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()

        # 自定义起始时间，减少生成 ID 的数字长度。
        self.epoch = 1704067200000
        self.worker_id_bits = 10
        self.sequence_bits = 12
        self.max_sequence = (1 << self.sequence_bits) - 1
        self.worker_id_shift = self.sequence_bits
        self.timestamp_shift = self.sequence_bits + self.worker_id_bits

    def next_id(self) -> int:
        # 加锁是为了保证同一个进程内并发生成 ID 时序列号安全递增。
        with self.lock:
            current_timestamp = self.current_millis()

            if current_timestamp < self.last_timestamp:
                raise RuntimeError("系统时钟回拨，暂时无法生成雪花 ID")

            if current_timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.max_sequence
                if self.sequence == 0:
                    current_timestamp = self.wait_next_millis(self.last_timestamp)
            else:
                self.sequence = 0

            self.last_timestamp = current_timestamp
            return (
                ((current_timestamp - self.epoch) << self.timestamp_shift)
                | (self.worker_id << self.worker_id_shift)
                | self.sequence
            )

    def current_millis(self) -> int:
        return int(time.time() * 1000)

    def wait_next_millis(self, last_timestamp: int) -> int:
        timestamp = self.current_millis()
        while timestamp <= last_timestamp:
            timestamp = self.current_millis()
        return timestamp


snowflake_generator = SnowflakeIdGenerator(worker_id=1)


def next_snowflake_id() -> str:
    return str(snowflake_generator.next_id())
