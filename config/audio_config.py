"""
Audio Configuration - Audio system specific settings and device management
"""

import platform
from typing import Dict, List, Optional, Tuple
import json

from utils.logger import Logger

class AudioConfig:
    """Audio configuration and device management"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger()
        
        # Platform-specific settings
        self.platform = platform.system()
        
        # Audio driver preferences by platform
        self.driver_preferences = {
            'Windows': ['WASAPI', 'DirectSound', 'MME'],
            'Darwin': ['CoreAudio'],  # macOS
            'Linux': ['ALSA', 'PulseAudio', 'JACK'],
        }
        
        # Recommended settings by platform
        self.platform_defaults = {
            'Windows': {
                'sample_rate': 44100,
                'buffer_size': 512,
                'exclusive_mode': False,
                'enable_input_monitoring': True,
            },
            'Darwin': {  # macOS
                'sample_rate': 44100,
                'buffer_size': 256,
                'exclusive_mode': False,
                'enable_input_monitoring': True,
            },
            'Linux': {
                'sample_rate': 44100,
                'buffer_size': 512,
                'exclusive_mode': False,
                'enable_input_monitoring': True,
            },
        }
        
        # Device quality ratings (for automatic selection)
        self.device_preferences = {
            # Professional audio interfaces (high quality)
            'focusrite': 10,
            'scarlett': 10,
            'presonus': 9,
            'audiobox': 9,
            'behringer': 8,
            'motu': 10,
            'rme': 10,
            'universal audio': 10,
            'steinberg': 9,
            'ur22': 9,
            'ur44': 9,
            'zoom': 7,
            'tascam': 7,
            'roland': 8,
            'boss': 8,
            
            # Consumer audio devices (medium quality)
            'realtek': 5,
            'conexant': 5,
            'via': 5,
            'creative': 6,
            'sound blaster': 6,
            
            # Built-in audio (lower quality for music production)
            'built-in': 3,
            'internal': 3,
            'integrated': 3,
            'onboard': 3,
            'motherboard': 3,
        }
        
        # Initialize platform-specific defaults
        self._apply_platform_defaults()
    
    def _apply_platform_defaults(self):
        """Apply platform-specific default settings"""
        try:
            platform_settings = self.platform_defaults.get(self.platform, {})
            
            for key, value in platform_settings.items():
                setting_key = f'audio.{key}'
                if self.settings.get(setting_key) is None:
                    self.settings.set(setting_key, value)
                    self.logger.debug(f"Applied platform default: {setting_key} = {value}")
        
        except Exception as e:
            self.logger.error(f"Error applying platform defaults: {e}")
    
    def get_recommended_buffer_size(self, device_name: str = "") -> int:
        """Get recommended buffer size for a device"""
        try:
            # Professional interfaces can handle smaller buffers
            device_lower = device_name.lower()
            
            # Check for professional audio interfaces
            professional_keywords = ['focusrite', 'scarlett', 'presonus', 'motu', 'rme', 'universal audio']
            if any(keyword in device_lower for keyword in professional_keywords):
                return 128  # Smaller buffer for professional interfaces
            
            # USB audio interfaces
            usb_keywords = ['usb', 'audio interface', 'audiobox']
            if any(keyword in device_lower for keyword in usb_keywords):
                return 256
            
            # Built-in audio typically needs larger buffers
            builtin_keywords = ['built-in', 'internal', 'realtek', 'conexant']
            if any(keyword in device_lower for keyword in builtin_keywords):
                return 1024
            
            # Default buffer size
            return 512
            
        except Exception as e:
            self.logger.error(f"Error getting recommended buffer size: {e}")
            return 512
    
    def get_recommended_sample_rate(self, device_name: str = "") -> int:
        """Get recommended sample rate for a device"""
        try:
            # Most devices work well with 44.1kHz
            # Some professional interfaces prefer 48kHz
            device_lower = device_name.lower()
            
            professional_keywords = ['motu', 'rme', 'apogee', 'lynx']
            if any(keyword in device_lower for keyword in professional_keywords):
                return 48000
            
            return 44100
            
        except Exception as e:
            self.logger.error(f"Error getting recommended sample rate: {e}")
            return 44100
    
    def rate_device_quality(self, device_name: str) -> int:
        """Rate device quality for automatic selection (1-10 scale)"""
        try:
            device_lower = device_name.lower()
            
            # Check against known device preferences
            for keyword, rating in self.device_preferences.items():
                if keyword in device_lower:
                    return rating
            
            # Default rating for unknown devices
            return 5
            
        except Exception as e:
            self.logger.error(f"Error rating device quality: {e}")
            return 5
    
    def select_best_devices(self, input_devices: List[Dict], output_devices: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
        """Automatically select the best input and output devices"""
        try:
            best_input = None
            best_output = None
            best_input_score = 0
            best_output_score = 0
            
            # Rate and select best input device
            for i, device in enumerate(input_devices):
                score = self.rate_device_quality(device['name'])
                
                # Bonus for professional audio interfaces
                if 'interface' in device['name'].lower():
                    score += 2
                
                # Bonus for USB devices (usually external interfaces)
                if 'usb' in device['name'].lower():
                    score += 1
                
                if score > best_input_score:
                    best_input_score = score
                    best_input = i
            
            # Rate and select best output device
            for i, device in enumerate(output_devices):
                score = self.rate_device_quality(device['name'])
                
                # Bonus for professional audio interfaces
                if 'interface' in device['name'].lower():
                    score += 2
                
                # Bonus for USB devices
                if 'usb' in device['name'].lower():
                    score += 1
                
                if score > best_output_score:
                    best_output_score = score
                    best_output = i
            
            if best_input is not None:
                self.logger.info(f"Selected best input device: {input_devices[best_input]['name']} (score: {best_input_score})")
            
            if best_output is not None:
                self.logger.info(f"Selected best output device: {output_devices[best_output]['name']} (score: {best_output_score})")
            
            return best_input, best_output
            
        except Exception as e:
            self.logger.error(f"Error selecting best devices: {e}")
            return None, None
    
    def get_latency_settings(self, device_name: str = "") -> Dict[str, int]:
        """Get recommended latency settings for a device"""
        try:
            device_lower = device_name.lower()
            
            # Professional interfaces can handle lower latency
            professional_keywords = ['focusrite', 'scarlett', 'presonus', 'motu', 'rme']
            if any(keyword in device_lower for keyword in professional_keywords):
                return {
                    'buffer_size': 128,
                    'periods': 2,
                    'target_latency_ms': 6,  # ~6ms round-trip
                }
            
            # Semi-professional USB interfaces
            usb_interface_keywords = ['audiobox', 'ur22', 'ur44', 'zoom', 'tascam']
            if any(keyword in device_lower for keyword in usb_interface_keywords):
                return {
                    'buffer_size': 256,
                    'periods': 2,
                    'target_latency_ms': 12,  # ~12ms round-trip
                }
            
            # Consumer audio devices
            consumer_keywords = ['realtek', 'conexant', 'creative', 'built-in']
            if any(keyword in device_lower for keyword in consumer_keywords):
                return {
                    'buffer_size': 1024,
                    'periods': 3,
                    'target_latency_ms': 46,  # ~46ms round-trip
                }
            
            # Default settings
            return {
                'buffer_size': 512,
                'periods': 2,
                'target_latency_ms': 23,  # ~23ms round-trip
            }
            
        except Exception as e:
            self.logger.error(f"Error getting latency settings: {e}")
            return {
                'buffer_size': 512,
                'periods': 2,
                'target_latency_ms': 23,
            }
    
    def validate_audio_settings(self, sample_rate: int, buffer_size: int, channels: int = 1) -> bool:
        """Validate audio settings for compatibility"""
        try:
            # Validate sample rate
            valid_sample_rates = [22050, 44100, 48000, 88200, 96000]
            if sample_rate not in valid_sample_rates:
                self.logger.warning(f"Unusual sample rate: {sample_rate}")
                return False
            
            # Validate buffer size (should be power of 2)
            if buffer_size <= 0 or (buffer_size & (buffer_size - 1)) != 0:
                self.logger.warning(f"Buffer size should be power of 2: {buffer_size}")
                return False
            
            # Validate buffer size range
            if buffer_size < 64 or buffer_size > 8192:
                self.logger.warning(f"Buffer size out of recommended range: {buffer_size}")
                return False
            
            # Validate channels
            if channels < 1 or channels > 2:
                self.logger.warning(f"Invalid channel count: {channels}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating audio settings: {e}")
            return False
    
    def get_platform_audio_info(self) -> Dict[str, any]:
        """Get platform-specific audio system information"""
        try:
            info = {
                'platform': self.platform,
                'preferred_drivers': self.driver_preferences.get(self.platform, []),
                'recommended_settings': self.platform_defaults.get(self.platform, {}),
            }
            
            # Add platform-specific details
            if self.platform == 'Windows':
                info['notes'] = [
                    "WASAPI is preferred for low latency",
                    "Exclusive mode may reduce latency but limits other applications",
                    "Consider ASIO drivers for professional interfaces"
                ]
            elif self.platform == 'Darwin':  # macOS
                info['notes'] = [
                    "CoreAudio provides excellent low-latency performance",
                    "Built-in audio interfaces are generally high quality",
                    "Professional interfaces work well with default drivers"
                ]
            elif self.platform == 'Linux':
                info['notes'] = [
                    "JACK provides the lowest latency for professional use",
                    "PulseAudio is easier to configure but higher latency",
                    "ALSA direct access provides good performance"
                ]
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting platform audio info: {e}")
            return {
                'platform': 'unknown',
                'preferred_drivers': [],
                'recommended_settings': {},
                'notes': []
            }
    
    def optimize_settings_for_device(self, device_info: Dict) -> Dict[str, any]:
        """Optimize audio settings for a specific device"""
        try:
            device_name = device_info.get('name', '')
            
            # Get recommended settings
            buffer_size = self.get_recommended_buffer_size(device_name)
            sample_rate = self.get_recommended_sample_rate(device_name)
            latency_settings = self.get_latency_settings(device_name)
            
            optimized_settings = {
                'sample_rate': sample_rate,
                'buffer_size': buffer_size,
                'channels': 1,  # Mono for guitar input
                **latency_settings
            }
            
            # Add device-specific optimizations
            device_lower = device_name.lower()
            
            # Focusrite Scarlett specific optimizations
            if 'scarlett' in device_lower:
                optimized_settings.update({
                    'buffer_size': 128,
                    'sample_rate': 48000,
                    'exclusive_mode': True,
                    'notes': ['Enable direct monitoring on the interface', 
                             'Use Focusrite Control for optimal settings']
                })
            
            # PreSonus AudioBox optimizations
            elif 'audiobox' in device_lower:
                optimized_settings.update({
                    'buffer_size': 256,
                    'sample_rate': 44100,
                    'notes': ['Use Universal Control for driver settings',
                             'Enable zero-latency monitoring']
                })
            
            # Behringer interfaces
            elif 'behringer' in device_lower:
                optimized_settings.update({
                    'buffer_size': 512,  # Behringer typically needs larger buffers
                    'notes': ['Install latest ASIO drivers for best performance']
                })
            
            self.logger.info(f"Optimized settings for {device_name}: {optimized_settings}")
            return optimized_settings
            
        except Exception as e:
            self.logger.error(f"Error optimizing settings for device: {e}")
            return {
                'sample_rate': 44100,
                'buffer_size': 512,
                'channels': 1
            }
    
    def save_device_profile(self, device_name: str, settings: Dict):
        """Save optimized settings for a device"""
        try:
            profiles_key = 'audio.device_profiles'
            profiles = self.settings.get(profiles_key, {})
            
            profiles[device_name] = {
                'settings': settings,
                'last_used': time.time(),
                'platform': self.platform
            }
            
            self.settings.set(profiles_key, profiles)
            self.logger.info(f"Saved device profile for: {device_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving device profile: {e}")
    
    def load_device_profile(self, device_name: str) -> Optional[Dict]:
        """Load saved settings for a device"""
        try:
            profiles_key = 'audio.device_profiles'
            profiles = self.settings.get(profiles_key, {})
            
            if device_name in profiles:
                profile = profiles[device_name]
                self.logger.info(f"Loaded device profile for: {device_name}")
                return profile.get('settings', {})
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading device profile: {e}")
            return None
