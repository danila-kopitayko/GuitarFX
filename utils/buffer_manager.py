"""
Buffer Manager - Manages audio buffers for real-time processing
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import threading

from utils.logger import Logger

class BufferManager:
    """Manages audio buffers for analysis and processing with thread safety"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger()
        
        # Buffer configuration
        self.analysis_window_size = settings.get('buffer.analysis_window_size', 4096)
        self.overlap_factor = settings.get('buffer.overlap_factor', 0.5)
        self.max_frames = settings.get('buffer.max_frames', 10)
        
        # Calculate overlap parameters
        self.hop_size = int(self.analysis_window_size * (1.0 - self.overlap_factor))
        
        # Audio buffers
        self.input_buffer = deque(maxlen=self.analysis_window_size * 2)  # Circular buffer for input
        self.frame_buffer = deque(maxlen=self.max_frames)  # Buffer for complete frames
        
        # Window function for analysis
        self.analysis_window = np.hanning(self.analysis_window_size)
        
        # Threading lock for buffer access
        self.buffer_lock = threading.Lock()
        
        # Buffer statistics
        self.frames_added = 0
        self.frames_dropped = 0
        self.windows_extracted = 0
        self._last_underrun_log = 0
        
        self.logger.debug(f"BufferManager initialized - window: {self.analysis_window_size}, hop: {self.hop_size}")
    
    def add_frame(self, audio_frame: np.ndarray):
        """Add a new audio frame to the buffer"""
        try:
            with self.buffer_lock:
                # Add samples to input buffer
                self.input_buffer.extend(audio_frame)
                
                # Add frame to frame buffer for processing
                if len(self.frame_buffer) >= self.max_frames:
                    # Drop oldest frame if buffer is full
                    self.frame_buffer.popleft()
                    self.frames_dropped += 1
                
                self.frame_buffer.append(audio_frame.copy())
                self.frames_added += 1
                
        except Exception as e:
            self.logger.error(f"Error adding frame to buffer: {e}")
    
    def has_complete_window(self) -> bool:
        """Check if we have enough samples for a complete analysis window"""
        with self.buffer_lock:
            return len(self.input_buffer) >= self.analysis_window_size
    
    def get_analysis_window(self) -> Optional[np.ndarray]:
        """Get a windowed audio segment for analysis"""
        try:
            with self.buffer_lock:
                if len(self.input_buffer) < self.analysis_window_size:
                    return None
                
                # Extract the most recent analysis window
                recent_samples = list(self.input_buffer)[-self.analysis_window_size:]
                analysis_data = np.array(recent_samples, dtype=np.float32)
                
                # Apply window function
                windowed_data = analysis_data * self.analysis_window
                
                self.windows_extracted += 1
                
                return windowed_data
                
        except Exception as e:
            self.logger.error(f"Error getting analysis window: {e}")
            return None
    
    def get_overlapping_windows(self, num_windows: int = 1) -> list:
        """Get multiple overlapping analysis windows"""
        try:
            windows = []
            
            with self.buffer_lock:
                if len(self.input_buffer) < self.analysis_window_size:
                    return windows
                
                buffer_array = np.array(list(self.input_buffer), dtype=np.float32)
                
                # Extract overlapping windows
                for i in range(num_windows):
                    start_idx = len(buffer_array) - self.analysis_window_size - (i * self.hop_size)
                    end_idx = start_idx + self.analysis_window_size
                    
                    if start_idx < 0:
                        break
                    
                    # Extract and window the segment
                    segment = buffer_array[start_idx:end_idx]
                    windowed_segment = segment * self.analysis_window
                    windows.insert(0, windowed_segment)  # Insert at beginning to maintain time order
                
                self.windows_extracted += len(windows)
                
            return windows
            
        except Exception as e:
            self.logger.error(f"Error getting overlapping windows: {e}")
            return []
    
    def get_recent_frames(self, num_frames: int = 1) -> list:
        """Get the most recent audio frames"""
        try:
            with self.buffer_lock:
                if num_frames <= 0:
                    return []
                
                # Get the most recent frames
                recent_frames = list(self.frame_buffer)[-num_frames:]
                return [frame.copy() for frame in recent_frames]
                
        except Exception as e:
            self.logger.error(f"Error getting recent frames: {e}")
            return []
    
    def get_concatenated_audio(self, duration_seconds: float) -> Optional[np.ndarray]:
        """Get concatenated audio data for a specific duration"""
        try:
            sample_rate = self.settings.get('audio.sample_rate', 44100)
            required_samples = int(duration_seconds * sample_rate)
            
            with self.buffer_lock:
                if len(self.input_buffer) < required_samples:
                    return None
                
                # Get the most recent samples
                recent_samples = list(self.input_buffer)[-required_samples:]
                return np.array(recent_samples, dtype=np.float32)
                
        except Exception as e:
            self.logger.error(f"Error getting concatenated audio: {e}")
            return None
    
    def clear_buffers(self):
        """Clear all buffers"""
        try:
            with self.buffer_lock:
                self.input_buffer.clear()
                self.frame_buffer.clear()
                
                # Reset statistics
                self.frames_added = 0
                self.frames_dropped = 0
                self.windows_extracted = 0
                
            self.logger.debug("Buffers cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing buffers: {e}")
    
    def get_buffer_level(self) -> float:
        """Get current buffer level as percentage of maximum"""
        try:
            with self.buffer_lock:
                max_capacity = self.analysis_window_size * 2
                current_level = len(self.input_buffer)
                return (current_level / max_capacity) * 100.0
                
        except Exception as e:
            self.logger.error(f"Error getting buffer level: {e}")
            return 0.0
    
    def is_buffer_healthy(self) -> bool:
        """Check if buffer is in a healthy state"""
        try:
            buffer_level = self.get_buffer_level()
            
            # Buffer should not be too empty or too full
            if buffer_level < 10.0:
                # Only log underrun warnings occasionally to prevent spam
                import time
                current_time = time.time()
                if not hasattr(self, '_last_underrun_log') or current_time - self._last_underrun_log > 5.0:
                    self.logger.warning(f"Buffer underrun: {buffer_level:.1f}%")
                    self._last_underrun_log = current_time
                return False
            
            if buffer_level > 95.0:
                self.logger.warning(f"Buffer overrun: {buffer_level:.1f}%")
                return False
            
            # Check drop rate
            if self.frames_added > 0:
                drop_rate = (self.frames_dropped / self.frames_added) * 100.0
                if drop_rate > 10.0:  # More than 10% drops is problematic
                    self.logger.warning(f"High frame drop rate: {drop_rate:.1f}%")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking buffer health: {e}")
            return False
    
    def get_latency_estimate(self) -> float:
        """Estimate current processing latency in milliseconds"""
        try:
            sample_rate = self.settings.get('audio.sample_rate', 44100)
            buffer_size = self.settings.get('audio.buffer_size', 512)
            
            # Base latency from audio buffer
            base_latency_ms = (buffer_size / sample_rate) * 1000.0
            
            # Additional latency from analysis buffer
            with self.buffer_lock:
                analysis_latency_samples = len(self.input_buffer)
            
            analysis_latency_ms = (analysis_latency_samples / sample_rate) * 1000.0
            
            total_latency_ms = base_latency_ms + analysis_latency_ms
            
            return total_latency_ms
            
        except Exception as e:
            self.logger.error(f"Error estimating latency: {e}")
            return 0.0
    
    def optimize_buffer_size(self):
        """Optimize buffer sizes based on current performance"""
        try:
            # Check if we need to adjust buffer sizes
            drop_rate = 0.0
            if self.frames_added > 0:
                drop_rate = (self.frames_dropped / self.frames_added) * 100.0
            
            buffer_level = self.get_buffer_level()
            
            # If drop rate is high, consider increasing buffer size
            if drop_rate > 5.0 and self.analysis_window_size < 8192:
                new_window_size = min(self.analysis_window_size * 2, 8192)
                self.logger.info(f"Increasing analysis window size: {self.analysis_window_size} -> {new_window_size}")
                self._resize_analysis_window(new_window_size)
            
            # If buffer is consistently underutilized, consider reducing size for lower latency
            elif drop_rate < 1.0 and buffer_level < 30.0 and self.analysis_window_size > 1024:
                new_window_size = max(self.analysis_window_size // 2, 1024)
                self.logger.info(f"Decreasing analysis window size: {self.analysis_window_size} -> {new_window_size}")
                self._resize_analysis_window(new_window_size)
                
        except Exception as e:
            self.logger.error(f"Error optimizing buffer size: {e}")
    
    def _resize_analysis_window(self, new_size: int):
        """Resize the analysis window (internal method)"""
        try:
            with self.buffer_lock:
                self.analysis_window_size = new_size
                self.hop_size = int(new_size * (1.0 - self.overlap_factor))
                self.analysis_window = np.hanning(new_size)
                
                # Resize input buffer
                max_size = new_size * 2
                new_buffer = deque(maxlen=max_size)
                
                # Transfer existing data
                existing_data = list(self.input_buffer)
                if len(existing_data) > max_size:
                    existing_data = existing_data[-max_size:]
                
                new_buffer.extend(existing_data)
                self.input_buffer = new_buffer
                
            self.logger.debug(f"Resized analysis window to {new_size}")
            
        except Exception as e:
            self.logger.error(f"Error resizing analysis window: {e}")
    
    def get_status(self) -> dict:
        """Get comprehensive buffer status"""
        try:
            with self.buffer_lock:
                input_buffer_size = len(self.input_buffer)
                frame_buffer_size = len(self.frame_buffer)
            
            drop_rate = 0.0
            if self.frames_added > 0:
                drop_rate = (self.frames_dropped / self.frames_added) * 100.0
            
            return {
                'analysis_window_size': self.analysis_window_size,
                'hop_size': self.hop_size,
                'overlap_factor': self.overlap_factor,
                'input_buffer_size': input_buffer_size,
                'input_buffer_capacity': self.input_buffer.maxlen,
                'frame_buffer_size': frame_buffer_size,
                'frame_buffer_capacity': self.max_frames,
                'buffer_level_percent': self.get_buffer_level(),
                'frames_added': self.frames_added,
                'frames_dropped': self.frames_dropped,
                'drop_rate_percent': drop_rate,
                'windows_extracted': self.windows_extracted,
                'estimated_latency_ms': self.get_latency_estimate(),
                'buffer_healthy': self.is_buffer_healthy(),
            }
            
        except Exception as e:
            self.logger.error(f"Error getting buffer status: {e}")
            return {}
    
    def reset_statistics(self):
        """Reset buffer statistics"""
        try:
            with self.buffer_lock:
                self.frames_added = 0
                self.frames_dropped = 0
                self.windows_extracted = 0
                
            self.logger.debug("Buffer statistics reset")
            
        except Exception as e:
            self.logger.error(f"Error resetting statistics: {e}")
