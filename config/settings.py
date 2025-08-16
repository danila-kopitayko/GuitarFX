"""
Settings - Configuration management system
"""

import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

from utils.logger import Logger

class Settings:
    """Application settings management with persistence"""
    
    def __init__(self, config_file: str = "config.json"):
        self.logger = Logger()
        self.config_file = config_file
        self.config_dir = Path.home() / ".guitar_effects"
        self.config_path = self.config_dir / config_file
        
        # Default configuration
        self.defaults = {
            # Audio settings
            'audio.sample_rate': 44100,
            'audio.buffer_size': 512,
            'audio.channels': 1,
            'audio.input_device': None,
            'audio.output_device': None,
            
            # Detection settings
            'detection.window_size': 2048,
            'detection.min_confidence': 0.7,
            'detection.confidence_threshold': 0.7,
            'detection.model_path': 'models/technique_model.pkl',
            'detection.scaler_path': 'models/technique_scaler.pkl',
            
            # Feature extraction settings
            'features.n_mfcc': 13,
            'features.n_fft': 2048,
            'features.hop_length': 512,
            
            # Effects settings
            'effects.crossfade_enabled': True,
            'effects.crossfade_time': 0.05,
            'effects.distortion': {
                'drive': 0.7,
                'tone': 0.6,
                'output_gain': 0.8,
                'bit_crush': 0.3,
                'saturation': 0.5,
                'low_cut': 80,
                'high_cut': 8000,
            },
            'effects.broken_record': {
                'scratch_intensity': 0.7,
                'wow_flutter': 0.5,
                'crackle': 0.4,
                'skip_probability': 0.1,
                'reverse_probability': 0.05,
                'pitch_bend': 0.6,
                'output_gain': 0.8,
                'vintage_filter': 0.5,
            },
            
            # GUI settings
            'gui.window_width': 1000,
            'gui.window_height': 700,
            'gui.theme': 'clam',
            'gui.update_rate': 10,  # Hz
            
            # Visualizer settings
            'visualizer.update_rate': 30,  # Hz
            'visualizer.history_length': 200,
            'visualizer.enabled': True,
            
            # Buffer management
            'buffer.analysis_window_size': 4096,
            'buffer.overlap_factor': 0.5,
            'buffer.max_frames': 10,
            
            # Logging settings
            'logging.level': 'INFO',
            'logging.file_enabled': True,
            'logging.console_enabled': True,
            'logging.max_file_size': 10485760,  # 10MB
            'logging.backup_count': 3,
        }
        
        # Current configuration
        self.config = {}
        
        # Load configuration
        self.load()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            # Try to get from current config first
            if key in self.config:
                return self.config[key]
            
            # Fall back to defaults
            if key in self.defaults:
                return self.defaults[key]
            
            # Return provided default or None
            return default
            
        except Exception as e:
            self.logger.error(f"Error getting config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        try:
            self.config[key] = value
            self.save()  # Auto-save on changes
            self.logger.debug(f"Set config: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error setting config key '{key}': {e}")
    
    def get_section(self, section_prefix: str) -> Dict[str, Any]:
        """Get all configuration values for a section"""
        section_config = {}
        
        # Get from current config
        for key, value in self.config.items():
            if key.startswith(section_prefix + '.'):
                section_key = key[len(section_prefix) + 1:]
                section_config[section_key] = value
        
        # Fill in defaults for missing keys
        for key, value in self.defaults.items():
            if key.startswith(section_prefix + '.'):
                section_key = key[len(section_prefix) + 1:]
                if section_key not in section_config:
                    section_config[section_key] = value
        
        return section_config
    
    def set_section(self, section_prefix: str, section_config: Dict[str, Any]):
        """Set all values for a configuration section"""
        try:
            for section_key, value in section_config.items():
                full_key = f"{section_prefix}.{section_key}"
                self.config[full_key] = value
            
            self.save()
            self.logger.debug(f"Set config section: {section_prefix}")
            
        except Exception as e:
            self.logger.error(f"Error setting config section '{section_prefix}': {e}")
    
    def load(self):
        """Load configuration from file"""
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Load configuration if file exists
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            else:
                # Create default configuration file
                self.config = self.defaults.copy()
                self.save()
                self.logger.info(f"Created default configuration at {self.config_path}")
            
            # Validate and update configuration
            self._validate_config()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Fall back to defaults
            self.config = self.defaults.copy()
    
    def save(self):
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.debug(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def _validate_config(self):
        """Validate and update configuration with any missing defaults"""
        updated = False
        
        # Add any missing default keys
        for key, default_value in self.defaults.items():
            if key not in self.config:
                self.config[key] = default_value
                updated = True
                self.logger.debug(f"Added missing config key: {key} = {default_value}")
        
        # Validate data types and ranges
        self._validate_audio_settings()
        self._validate_detection_settings()
        self._validate_effect_settings()
        
        # Save if we made any updates
        if updated:
            self.save()
    
    def _validate_audio_settings(self):
        """Validate audio-related settings"""
        # Sample rate validation
        sample_rate = self.get('audio.sample_rate')
        if sample_rate not in [22050, 44100, 48000, 96000]:
            self.logger.warning(f"Invalid sample rate {sample_rate}, using 44100")
            self.set('audio.sample_rate', 44100)
        
        # Buffer size validation
        buffer_size = self.get('audio.buffer_size')
        valid_buffer_sizes = [128, 256, 512, 1024, 2048, 4096]
        if buffer_size not in valid_buffer_sizes:
            self.logger.warning(f"Invalid buffer size {buffer_size}, using 512")
            self.set('audio.buffer_size', 512)
        
        # Channels validation
        channels = self.get('audio.channels')
        if not isinstance(channels, int) or channels < 1 or channels > 2:
            self.logger.warning(f"Invalid channels {channels}, using 1")
            self.set('audio.channels', 1)
    
    def _validate_detection_settings(self):
        """Validate detection-related settings"""
        # Confidence threshold validation
        confidence = self.get('detection.min_confidence')
        if not isinstance(confidence, (int, float)) or not 0.1 <= confidence <= 0.99:
            self.logger.warning(f"Invalid confidence threshold {confidence}, using 0.7")
            self.set('detection.min_confidence', 0.7)
        
        # Window size validation
        window_size = self.get('detection.window_size')
        if not isinstance(window_size, int) or window_size < 512 or window_size > 8192:
            self.logger.warning(f"Invalid window size {window_size}, using 2048")
            self.set('detection.window_size', 2048)
    
    def _validate_effect_settings(self):
        """Validate effect-related settings"""
        # Crossfade time validation
        crossfade_time = self.get('effects.crossfade_time')
        if not isinstance(crossfade_time, (int, float)) or not 0.001 <= crossfade_time <= 1.0:
            self.logger.warning(f"Invalid crossfade time {crossfade_time}, using 0.05")
            self.set('effects.crossfade_time', 0.05)
        
        # Validate effect parameter ranges
        self._validate_effect_params('effects.distortion')
        self._validate_effect_params('effects.broken_record')
    
    def _validate_effect_params(self, effect_key: str):
        """Validate effect parameter ranges"""
        effect_params = self.get(effect_key, {})
        if not isinstance(effect_params, dict):
            return
        
        # Define parameter ranges
        param_ranges = {
            'drive': (0.0, 1.0),
            'tone': (0.0, 1.0),
            'output_gain': (0.0, 1.0),
            'bit_crush': (0.0, 1.0),
            'saturation': (0.0, 1.0),
            'scratch_intensity': (0.0, 1.0),
            'wow_flutter': (0.0, 1.0),
            'crackle': (0.0, 1.0),
            'skip_probability': (0.0, 1.0),
            'reverse_probability': (0.0, 1.0),
            'pitch_bend': (0.0, 1.0),
            'vintage_filter': (0.0, 1.0),
            'low_cut': (20, 500),
            'high_cut': (1000, 20000),
        }
        
        # Validate each parameter
        for param_name, value in effect_params.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                if not isinstance(value, (int, float)) or not min_val <= value <= max_val:
                    default_value = self.defaults.get(effect_key, {}).get(param_name, min_val)
                    self.logger.warning(f"Invalid {effect_key}.{param_name} = {value}, using {default_value}")
                    effect_params[param_name] = default_value
        
        # Update the effect configuration
        self.set(effect_key, effect_params)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        try:
            self.config = self.defaults.copy()
            self.save()
            self.logger.info("Configuration reset to defaults")
            
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {e}")
    
    def reset_section(self, section_prefix: str):
        """Reset a configuration section to defaults"""
        try:
            # Remove all keys from the section
            keys_to_remove = [key for key in self.config.keys() if key.startswith(section_prefix + '.')]
            for key in keys_to_remove:
                del self.config[key]
            
            # Add back default values
            for key, value in self.defaults.items():
                if key.startswith(section_prefix + '.'):
                    self.config[key] = value
            
            self.save()
            self.logger.info(f"Reset config section: {section_prefix}")
            
        except Exception as e:
            self.logger.error(f"Error resetting config section '{section_prefix}': {e}")
    
    def export_config(self, export_path: str):
        """Export configuration to a file"""
        try:
            with open(export_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, import_path: str):
        """Import configuration from a file"""
        try:
            with open(import_path, 'r') as f:
                imported_config = json.load(f)
            
            # Validate imported configuration
            if isinstance(imported_config, dict):
                self.config = imported_config
                self._validate_config()
                self.save()
                self.logger.info(f"Configuration imported from {import_path}")
                return True
            else:
                self.logger.error("Invalid configuration format in import file")
                return False
                
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        # Merge defaults with current config
        all_config = self.defaults.copy()
        all_config.update(self.config)
        return all_config
    
    def get_config_path(self) -> Path:
        """Get the path to the configuration file"""
        return self.config_path
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the configuration"""
        return {
            'config_file': str(self.config_path),
            'config_exists': self.config_path.exists(),
            'config_dir': str(self.config_dir),
            'total_settings': len(self.config),
            'defaults_count': len(self.defaults),
        }
