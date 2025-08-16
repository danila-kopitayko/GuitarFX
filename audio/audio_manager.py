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
        self.test_mode = False
        self.test_thread = None
        
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
                
                if device_info.get('maxInputChannels', 0) > 0:
                    self.input_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
                
                if device_info.get('maxOutputChannels', 0) > 0:
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
            
            # Start input stream - handle device availability gracefully
            try:
                self.input_stream = self.pa.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=input_device,
                    frames_per_buffer=self.buffer_size,
                    stream_callback=self._input_callback
                )
                self.logger.info(f"Input stream started on device {input_device}")
            except Exception as e:
                self.logger.warning(f"Could not open input device {input_device}: {e}")
                self.logger.info("Starting in test mode with simulated guitar input")
                self._start_test_input_stream()
            
            # Start output stream
            try:
                self.output_stream = self.pa.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=output_device,
                    frames_per_buffer=self.buffer_size,
                    stream_callback=self._output_callback
                )
                self.logger.info(f"Output stream started on device {output_device}")
            except Exception as e:
                self.logger.warning(f"Could not open output device {output_device}: {e}")
                self.logger.info("Audio output not available - running in input-only mode")
            
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
            
            # Stop test input if running
            if hasattr(self, 'test_thread') and self.test_thread and self.test_thread.is_alive():
                self.test_thread.join(timeout=1.0)
            
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
    
    def _start_test_input_stream(self):
        """Start a test input stream with simulated guitar signal"""
        import threading
        import math
        
        self.test_mode = True
        self.test_thread = threading.Thread(target=self._test_input_generator, daemon=True)
        self.test_thread.start()
    
    def _test_input_generator(self):
        """Generate test guitar-like signals for development/testing"""
        import time
        
        sample_count = 0
        test_techniques = ['none', 'chugging', 'harmonic']
        current_technique = 0
        technique_duration = 3.0  # seconds per technique
        samples_per_technique = int(technique_duration * self.sample_rate)
        
        while self.stream_active:
            try:
                # Generate a buffer worth of samples
                audio_data = np.zeros(self.buffer_size, dtype=np.float32)
                
                for i in range(self.buffer_size):
                    t = sample_count / self.sample_rate
                    
                    # Switch techniques periodically
                    technique_index = (sample_count // samples_per_technique) % len(test_techniques)
                    technique = test_techniques[technique_index]
                    
                    # Generate different signals for each technique
                    if technique == 'chugging':
                        # Palm-muted low frequency chugging
                        freq = 82.4  # Low E string
                        signal = np.sin(2 * np.pi * freq * t) * 0.3
                        # Add aggressive envelope
                        envelope = max(0, 1 - (t % 0.2) * 5)  # Sharp attack, quick decay
                        signal *= envelope
                        # Add some harmonic distortion
                        signal += 0.1 * np.sin(6 * np.pi * freq * t) * envelope
                        
                    elif technique == 'harmonic':
                        # Pinch harmonics - high frequency content
                        fundamental = 82.4
                        # Emphasize 12th fret harmonic (octave)
                        signal = 0.2 * np.sin(2 * np.pi * fundamental * 2 * t)
                        # Add 5th harmonic for pinch harmonic character
                        signal += 0.4 * np.sin(2 * np.pi * fundamental * 5 * t)
                        # Quick decay envelope
                        envelope = max(0, np.exp(-t % 1.0 * 3))
                        signal *= envelope
                        
                    else:  # 'none' - clean guitar
                        # Clean single notes or chords
                        freq = 196.0  # G3
                        signal = 0.4 * np.sin(2 * np.pi * freq * t)
                        # Natural guitar decay
                        envelope = max(0, np.exp(-t % 2.0 * 1.5))
                        signal *= envelope
                    
                    # Add some noise to simulate real guitar pickup
                    noise = np.random.normal(0, 0.02)
                    audio_data[i] = signal + noise
                    
                    sample_count += 1
                
                # Put the generated data into the input buffer
                try:
                    self.input_buffer.put_nowait(audio_data)
                except:
                    # Drop if buffer full
                    try:
                        self.input_buffer.get_nowait()
                        self.input_buffer.put_nowait(audio_data)
                    except:
                        pass
                
                # Sleep to match real-time audio rate
                time.sleep(self.buffer_size / self.sample_rate)
                
            except Exception as e:
                self.logger.error(f"Test input generator error: {e}")
                time.sleep(0.1)
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            self.stop_stream()
            self.pa.terminate()
            self.logger.info("Audio manager cleanup complete")
        except Exception as e:
            self.logger.error(f"Audio cleanup error: {e}")
