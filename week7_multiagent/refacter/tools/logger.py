"""
日志工具 - 将日志输出到文件和控制台

功能：
1. 将日志同时输出到控制台和文件
2. 支持不同日志级别（INFO, DEBUG, WARNING, ERROR）
3. 日志文件自动按日期命名
4. 支持自定义输出目录
"""
import os
import sys
import time
from datetime import datetime
from typing import Optional


class AgentLogger:
    """日志记录器"""
    
    def __init__(self, log_dir: str = "output", log_level: str = "INFO"):
        """
        Args:
            log_dir: 日志输出目录
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = log_dir
        self.log_level = log_level.upper()
        
        # 确保输出目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 获取今日日期作为日志文件名
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = os.path.join(log_dir, f"agent_{today}.log")
        
        # 日志级别映射
        self.level_priority = {
            "DEBUG": 1,
            "INFO": 2,
            "WARNING": 3,
            "ERROR": 4
        }
    
    def _should_log(self, level: str) -> bool:
        """检查是否应该记录该级别的日志"""
        return self.level_priority.get(level.upper(), 2) >= self.level_priority.get(self.log_level, 2)
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _log(self, level: str, message: str):
        """记录日志"""
        if not self._should_log(level):
            return
        
        timestamp = self._get_timestamp()
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        # 输出到控制台
        print(f"[{level}] {message}")
        
        # 输出到文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            print(f"[ERROR] 写入日志文件失败: {str(e)}")
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self._log("DEBUG", message)
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self._log("INFO", message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self._log("WARNING", message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self._log("ERROR", message)
    
    def log_tool_call(self, tool_name: str, arguments: dict, result: str):
        """记录工具调用日志"""
        self.info(f"🔧 执行工具: {tool_name}")
        # 记录完整参数（包括代码内容）
        for key, value in arguments.items():
            if key == 'code':
                # 代码内容单独记录，完整保存
                self.debug(f"📋 参数[{key}]:\n{value}")
            else:
                self.debug(f"📋 参数[{key}]: {value}")
        # 记录执行结果
        if "失败" in result or "错误" in result:
            self.error(f"❌ 工具调用失败: {tool_name}")
            self.error(f"📝 错误信息: {result}")
        else:
            self.info(f"✅ 工具调用成功: {tool_name}")
            self.debug(f"📝 执行结果: {result[:500]}..." if len(result) > 500 else f"📝 执行结果: {result}")
    
    def log_agent_step(self, step: int, action: str, content: str = ""):
        """记录Agent步骤日志"""
        self.info(f"📌 步骤 {step}: {action}")
        if content:
            self.debug(f"📄 内容: {content[:200]}..." if len(content) > 200 else f"📄 内容: {content}")
    
    def log_execution(self, agent_name: str, input_text: str, output_text: str):
        """记录执行日志"""
        self.info(f"🚀 {agent_name} 执行开始")
        self.debug(f"📥 输入: {input_text[:200]}..." if len(input_text) > 200 else f"📥 输入: {input_text}")
        self.debug(f"📤 输出: {output_text[:200]}..." if len(output_text) > 200 else f"📤 输出: {output_text}")
        self.info(f"✅ {agent_name} 执行完成")


# 创建全局日志实例（设置为DEBUG级别以记录详细信息）
logger = AgentLogger(log_level="DEBUG")


def get_logger() -> AgentLogger:
    """获取全局日志实例"""
    return logger
