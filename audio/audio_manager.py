"""
Audio Manager - Handles audio I/O operations using PyAudio
"""

import pyaudio
import numpy as np
import threading
import queue
from typing import List, Dict, Optional, Tuple

from utils.logger import Logger

class AudioManager:
    """Manages audio input/output streams and device configuration"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger()
        
        # PyAudio instance
        self.pa = pyaudio.PyAudio()
        
        # Stream configuration
        self.input_stream = None
        self.output_stream = None
        self.stream_active = False
        
        # Audio buffers
        self.input_buffer = queue.Queue(maxsize=10)
        self.output_buffer = queue.Queue(maxsize=10)
        
        # Device information
        self.input_devices = []
        self.output_devices = []
        
        # Current settings
        self.sample_rate = self.settings.get('audio.sample_rate', 44100)
        self.buffer_size = self.settings.get('audio.buffer_size', 512)
        self.channels = self.settings.get('audio.channels', 1)
        self.format = pyaudio.paFloat32
        
        # Initialize devices
        self._scan_devices()
        
    def _scan_devices(self):
        """Scan for available audio devices"""
        try:
            self.input_devices = []
            self.output_devices = []
            
            device_count = self.pa.get_device_count()
            
            for i in range(device_count):
                device_info = self.pa.get_device_info_by_index(i)
                
                if device_info['maxInputChannels'] > 0:
                    self.input_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
                
                if device_info['maxOutputChannels'] > 0:
                    self.output_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxOutputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
            
            self.logger.info(f"Found {len(self.input_devices)} input devices")
            self.logger.info(f"Found {len(self.output_devices)} output devices")
            
        except Exception as e:
            self.logger.error(f"Failed to scan audio devices: {e}")
    
    def get_input_devices(self) -> List[Dict]:
        """Get list of available input devices"""
        return self.input_devices.copy()
    
    def get_output_devices(self) -> List[Dict]:
        """Get list of available output devices"""
        return self.output_devices.copy()
    
    def set_input_device(self, device_index: int) -> bool:
        """Set the input device"""
        try:
            if device_index < len(self.input_devices):
                self.settings.set('audio.input_device', device_index)
                self.logger.info(f"Input device set to: {self.input_devices[device_index]['name']}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to set input device: {e}")
            return False
    
    def set_output_device(self, device_index: int) -> bool:
        """Set the output device"""
        try:
            if device_index < len(self.output_devices):
                self.settings.set('audio.output_device', device_index)
                self.logger.info(f"Output device set to: {self.output_devices[device_index]['name']}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to set output device: {e}")
            return False
    
    def _input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        try:
            if status:
                self.logger.warning(f"Input stream status: {status}")
            
            # Convert to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Put in input buffer (non-blocking)
            try:
                self.input_buffer.put_nowait(audio_data.copy())
            except queue.Full:
                # Drop oldest frame if buffer is full
                try:
                    self.input_buffer.get_nowait()
                    self.input_buffer.put_nowait(audio_data.copy())
                except queue.Empty:
                    pass
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            self.logger.error(f"Input callback error: {e}")
            return (None, pyaudio.paAbort)
    
    def _output_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio output stream"""
        try:
            if status:
                self.logger.warning(f"Output stream status: {status}")
            
            # Get processed audio from output buffer
            try:
                audio_data = self.output_buffer.get_nowait()
                
                # Ensure correct frame count
                if len(audio_data) != frame_count:
                    if len(audio_data) < frame_count:
                        # Pad with zeros
                        audio_data = np.pad(audio_data, (0, frame_count - len(audio_data)))
                    else:
                        # Truncate
                        audio_data = audio_data[:frame_count]
                
                return (audio_data.tobytes(), pyaudio.paContinue)
                
            except queue.Empty:
                # Return silence if no processed audio available
                silence = np.zeros(frame_count, dtype=np.float32)
                return (silence.tobytes(), pyaudio.paContinue)
            
        except Exception as e:
            self.logger.error(f"Output callback error: {e}")
            silence = np.zeros(frame_count, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)
    
    def start_stream(self) -> bool:
        """Start audio input/output streams"""
        try:
            if self.stream_active:
                return True
            
            # Get device indices
            input_device = self.settings.get('audio.input_device', None)
            output_device = self.settings.get('audio.output_device', None)
            
            # Start input stream
            self.input_stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._input_callback
            )
            
            # Start output stream
            self.output_stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=output_device,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._output_callback
            )
            
            self.stream_active = True
            self.logger.info("Audio streams started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start audio streams: {e}")
            self.stop_stream()
            return False
    
    def stop_stream(self):
        """Stop audio streams"""
        try:
            self.stream_active = False
            
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
            
            # Clear buffers
            while not self.input_buffer.empty():
                try:
                    self.input_buffer.get_nowait()
                except queue.Empty:
                    break
            
            while not self.output_buffer.empty():
                try:
                    self.output_buffer.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("Audio streams stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping audio streams: {e}")
    
    def get_input_frame(self) -> Optional[np.ndarray]:
        """Get the next input audio frame"""
        try:
            return self.input_buffer.get_nowait()
        except queue.Empty:
            return None
    
    def put_output_frame(self, audio_data: np.ndarray):
        """Put processed audio frame to output buffer"""
        try:
            self.output_buffer.put_nowait(audio_data)
        except queue.Full:
            # Drop oldest frame if buffer is full
            try:
                self.output_buffer.get_nowait()
                self.output_buffer.put_nowait(audio_data)
            except queue.Empty:
                pass
    
    def is_stream_active(self) -> bool:
        """Check if audio streams are active"""
        return self.stream_active
    
    def get_stream_info(self) -> Dict:
        """Get current stream information"""
        return {
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size,
            'channels': self.channels,
            'format': 'float32',
            'active': self.stream_active,
            'input_latency': self.input_stream.get_input_latency() if self.input_stream else 0,
            'output_latency': self.output_stream.get_output_latency() if self.output_stream else 0
        }
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            self.stop_stream()
            self.pa.terminate()
            self.logger.info("Audio manager cleanup complete")
        except Exception as e:
            self.logger.error(f"Audio cleanup error: {e}")
