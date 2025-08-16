"""
Audio Processor - Main audio processing pipeline
"""

import numpy as np
import time
from typing import Optional

from utils.logger import Logger
from utils.buffer_manager import BufferManager

class AudioProcessor:
    """Main audio processing pipeline that coordinates detection and effects"""
    
    def __init__(self, audio_manager, technique_detector, effect_processor, settings):
        self.audio_manager = audio_manager
        self.technique_detector = technique_detector
        self.effect_processor = effect_processor
        self.settings = settings
        self.logger = Logger()
        
        # Processing state
        self.processing_enabled = False
        self.bypass_effects = False
        
        # Buffer management
        self.buffer_manager = BufferManager(settings)
        
        # Performance monitoring
        self.frame_count = 0
        self.processing_times = []
        self.last_stats_time = time.time()
        
        # Current technique state
        self.current_technique = None
        self.technique_confidence = 0.0
        
    def process_frame(self):
        """Process one audio frame through the complete pipeline"""
        start_time = time.time()
        
        try:
            # Get input audio frame
            input_frame = self.audio_manager.get_input_frame()
            if input_frame is None:
                return
            
            # Add to processing buffer
            self.buffer_manager.add_frame(input_frame)
            
            # Check if we have enough frames for processing
            if not self.buffer_manager.has_complete_window():
                # Pass through unprocessed audio for low latency
                self._pass_through_audio(input_frame)
                return
            
            # Get windowed audio for analysis
            analysis_window = self.buffer_manager.get_analysis_window()
            
            # Detect playing technique
            technique, confidence = self.technique_detector.detect_technique(analysis_window)
            
            # Update technique state
            self._update_technique_state(technique, confidence)
            
            # Process audio with effects based on detected technique
            if self.processing_enabled and not self.bypass_effects:
                processed_frame = self._apply_effects(input_frame)
            else:
                processed_frame = input_frame
            
            # Send to output
            self.audio_manager.put_output_frame(processed_frame)
            
            # Update performance stats
            self._update_performance_stats(start_time)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            # Pass through original audio on error
            if input_frame is not None:
                self._pass_through_audio(input_frame)
    
    def _pass_through_audio(self, audio_frame):
        """Pass audio through without processing"""
        self.audio_manager.put_output_frame(audio_frame)
    
    def _update_technique_state(self, technique, confidence):
        """Update the current technique state with hysteresis"""
        # Apply confidence threshold
        min_confidence = self.settings.get('detection.min_confidence', 0.7)
        
        if confidence >= min_confidence:
            # High confidence detection
            if self.current_technique != technique:
                self.logger.debug(f"Technique changed: {self.current_technique} -> {technique}")
                self.current_technique = technique
                self.technique_confidence = confidence
                
                # Notify effect processor of technique change
                self.effect_processor.set_technique(technique, confidence)
        else:
            # Low confidence - use hysteresis
            if self.technique_confidence > 0.5:
                self.technique_confidence *= 0.95  # Decay confidence
            else:
                # Clear technique if confidence too low
                if self.current_technique is not None:
                    self.logger.debug(f"Technique cleared: {self.current_technique}")
                    self.current_technique = None
                    self.technique_confidence = 0.0
                    self.effect_processor.set_technique(None, 0.0)
    
    def _apply_effects(self, audio_frame):
        """Apply effects based on current technique"""
        try:
            return self.effect_processor.process_audio(audio_frame)
        except Exception as e:
            self.logger.error(f"Effect processing error: {e}")
            return audio_frame
    
    def _update_performance_stats(self, start_time):
        """Update performance monitoring statistics"""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.frame_count += 1
        
        # Update stats every second
        current_time = time.time()
        if current_time - self.last_stats_time >= 1.0:
            self._log_performance_stats()
            self.last_stats_time = current_time
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        if not self.processing_times:
            return
        
        avg_time = np.mean(self.processing_times)
        max_time = np.max(self.processing_times)
        
        # Calculate CPU usage percentage
        buffer_size = self.settings.get('audio.buffer_size', 512)
        sample_rate = self.settings.get('audio.sample_rate', 44100)
        frame_duration = buffer_size / sample_rate
        cpu_usage = (avg_time / frame_duration) * 100
        
        if cpu_usage > 80:
            self.logger.warning(f"High CPU usage: {cpu_usage:.1f}%")
        else:
            self.logger.debug(f"Processing stats - Avg: {avg_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms, CPU: {cpu_usage:.1f}%")
        
        # Clear old stats
        self.processing_times = []
    
    def enable_processing(self):
        """Enable audio processing"""
        self.processing_enabled = True
        self.logger.info("Audio processing enabled")
    
    def disable_processing(self):
        """Disable audio processing (bypass mode)"""
        self.processing_enabled = False
        self.effect_processor.clear_all_effects()
        self.logger.info("Audio processing disabled")
    
    def set_bypass(self, bypass: bool):
        """Set effect bypass state"""
        self.bypass_effects = bypass
        if bypass:
            self.effect_processor.clear_all_effects()
        self.logger.info(f"Effects bypass: {bypass}")
    
    def get_current_technique(self) -> Optional[str]:
        """Get the currently detected technique"""
        return self.current_technique
    
    def get_technique_confidence(self) -> float:
        """Get the confidence of current technique detection"""
        return self.technique_confidence
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            'frame_count': self.frame_count,
            'processing_enabled': self.processing_enabled,
            'bypass_effects': self.bypass_effects,
            'current_technique': self.current_technique,
            'technique_confidence': self.technique_confidence,
            'buffer_status': self.buffer_manager.get_status()
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.processing_times = []
        self.last_stats_time = time.time()
        self.logger.info("Performance stats reset")
