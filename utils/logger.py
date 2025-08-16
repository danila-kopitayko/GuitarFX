"""
Logger - Centralized logging system for the application
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
import threading

class Logger:
    """Centralized logging system with file and console output"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single logger instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Logger configuration
        self.logger_name = "GuitarEffects"
        self.log_level = logging.INFO
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        self.date_format = "%Y-%m-%d %H:%M:%S"
        
        # File logging configuration
        self.log_dir = Path.home() / ".guitar_effects" / "logs"
        self.log_filename = "guitar_effects.log"
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 3
        
        # Console logging
        self.console_enabled = True
        self.file_enabled = True
        
        # Create logger
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Initial log message
        self.info("Logger initialized")
    
    def _setup_handlers(self):
        """Setup logging handlers for file and console output"""
        try:
            # Create formatter
            formatter = logging.Formatter(self.log_format, self.date_format)
            
            # Console handler
            if self.console_enabled:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.log_level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            # File handler
            if self.file_enabled:
                # Create log directory
                self.log_dir.mkdir(parents=True, exist_ok=True)
                
                # Create rotating file handler
                log_file_path = self.log_dir / self.log_filename
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Error setting up logger handlers: {e}", file=sys.stderr)
    
    def set_level(self, level):
        """Set logging level"""
        try:
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            
            self.log_level = level
            self.logger.setLevel(level)
            
            # Update handler levels
            for handler in self.logger.handlers:
                handler.setLevel(level)
            
            self.info(f"Log level set to: {logging.getLevelName(level)}")
            
        except Exception as e:
            print(f"Error setting log level: {e}", file=sys.stderr)
    
    def enable_file_logging(self, enabled: bool = True):
        """Enable or disable file logging"""
        try:
            self.file_enabled = enabled
            
            # Remove existing file handlers
            file_handlers = [h for h in self.logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
            for handler in file_handlers:
                self.logger.removeHandler(handler)
            
            # Add file handler if enabled
            if enabled:
                formatter = logging.Formatter(self.log_format, self.date_format)
                self.log_dir.mkdir(parents=True, exist_ok=True)
                
                log_file_path = self.log_dir / self.log_filename
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
            self.info(f"File logging {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            print(f"Error configuring file logging: {e}", file=sys.stderr)
    
    def enable_console_logging(self, enabled: bool = True):
        """Enable or disable console logging"""
        try:
            self.console_enabled = enabled
            
            # Remove existing console handlers
            console_handlers = [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler) 
                              and not isinstance(h, logging.handlers.RotatingFileHandler)]
            for handler in console_handlers:
                self.logger.removeHandler(handler)
            
            # Add console handler if enabled
            if enabled:
                formatter = logging.Formatter(self.log_format, self.date_format)
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.log_level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            self.info(f"Console logging {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            print(f"Error configuring console logging: {e}", file=sys.stderr)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception with traceback"""
        self.logger.exception(message)
    
    def log_function_entry(self, function_name: str, *args, **kwargs):
        """Log function entry for debugging"""
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        self.debug(f"Entering {function_name}({params})")
    
    def log_function_exit(self, function_name: str, result=None):
        """Log function exit for debugging"""
        if result is not None:
            self.debug(f"Exiting {function_name}, result: {result}")
        else:
            self.debug(f"Exiting {function_name}")
    
    def log_performance(self, operation: str, duration_ms: float, details: str = ""):
        """Log performance metrics"""
        message = f"Performance - {operation}: {duration_ms:.2f}ms"
        if details:
            message += f" ({details})"
        
        if duration_ms > 100:  # Log as warning if > 100ms
            self.warning(message)
        else:
            self.debug(message)
    
    def log_audio_stats(self, stats: dict):
        """Log audio processing statistics"""
        self.debug(f"Audio Stats - Frames: {stats.get('frame_count', 0)}, "
                  f"Drops: {stats.get('drops', 0)}, "
                  f"Latency: {stats.get('latency_ms', 0):.1f}ms")
    
    def log_technique_detection(self, technique: str, confidence: float):
        """Log technique detection results"""
        self.debug(f"Technique Detection - {technique}: {confidence:.2%} confidence")
    
    def log_effect_change(self, effect_name: str, parameter: str, value, old_value=None):
        """Log effect parameter changes"""
        if old_value is not None:
            self.debug(f"Effect Change - {effect_name}.{parameter}: {old_value} -> {value}")
        else:
            self.debug(f"Effect Set - {effect_name}.{parameter}: {value}")
    
    def get_log_file_path(self) -> Path:
        """Get the path to the current log file"""
        return self.log_dir / self.log_filename
    
    def get_log_files(self) -> list:
        """Get list of all log files"""
        try:
            if not self.log_dir.exists():
                return []
            
            log_files = []
            for file_path in self.log_dir.glob("*.log*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    log_files.append({
                        'path': file_path,
                        'name': file_path.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
            
            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: x['modified'], reverse=True)
            return log_files
            
        except Exception as e:
            self.error(f"Error getting log files: {e}")
            return []
    
    def clear_old_logs(self, days_to_keep: int = 7):
        """Clear log files older than specified days"""
        try:
            if not self.log_dir.exists():
                return 0
            
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            files_deleted = 0
            
            for file_path in self.log_dir.glob("*.log*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        files_deleted += 1
            
            if files_deleted > 0:
                self.info(f"Deleted {files_deleted} old log files")
            
            return files_deleted
            
        except Exception as e:
            self.error(f"Error clearing old logs: {e}")
            return 0
    
    def get_recent_logs(self, max_lines: int = 100) -> list:
        """Get recent log entries"""
        try:
            log_file_path = self.get_log_file_path()
            
            if not log_file_path.exists():
                return []
            
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Return the last max_lines
            return lines[-max_lines:] if lines else []
            
        except Exception as e:
            self.error(f"Error getting recent logs: {e}")
            return []
    
    def log_system_info(self):
        """Log system information"""
        try:
            import platform
            import psutil
            
            self.info("=== System Information ===")
            self.info(f"Platform: {platform.platform()}")
            self.info(f"Python: {platform.python_version()}")
            self.info(f"CPU: {platform.processor()}")
            self.info(f"CPU Count: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
            self.info(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
            self.info("=== End System Information ===")
            
        except ImportError:
            self.info("System information not available (psutil not installed)")
        except Exception as e:
            self.error(f"Error logging system info: {e}")
    
    def create_context_logger(self, context: str):
        """Create a context-specific logger"""
        return ContextLogger(self, context)

class ContextLogger:
    """Logger wrapper that adds context to all messages"""
    
    def __init__(self, main_logger: Logger, context: str):
        self.main_logger = main_logger
        self.context = context
    
    def _format_message(self, message: str) -> str:
        return f"[{self.context}] {message}"
    
    def debug(self, message: str):
        self.main_logger.debug(self._format_message(message))
    
    def info(self, message: str):
        self.main_logger.info(self._format_message(message))
    
    def warning(self, message: str):
        self.main_logger.warning(self._format_message(message))
    
    def error(self, message: str):
        self.main_logger.error(self._format_message(message))
    
    def critical(self, message: str):
        self.main_logger.critical(self._format_message(message))
    
    def exception(self, message: str):
        self.main_logger.exception(self._format_message(message))
