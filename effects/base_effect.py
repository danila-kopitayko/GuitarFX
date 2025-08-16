"""
Base Effect - Abstract base class for all audio effects
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseEffect(ABC):
    """Abstract base class for all audio effects"""
    
    def __init__(self, sample_rate: int, **kwargs):
        self.sample_rate = sample_rate
        self.enabled = False
        self.parameters = {}
        self.state = {}
        
        # Initialize effect-specific parameters
        self._initialize_parameters(**kwargs)
        
        # Initialize internal state
        self._initialize_state()
    
    @abstractmethod
    def _initialize_parameters(self, **kwargs):
        """Initialize effect-specific parameters"""
        pass
    
    @abstractmethod
    def _initialize_state(self):
        """Initialize internal effect state"""
        pass
    
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio data through the effect
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Processed audio samples
        """
        pass
    
    def enable(self):
        """Enable the effect"""
        self.enabled = True
        self.on_enable()
    
    def disable(self):
        """Disable the effect"""
        self.enabled = False
        self.on_disable()
    
    def on_enable(self):
        """Called when effect is enabled - override for custom behavior"""
        pass
    
    def on_disable(self):
        """Called when effect is disabled - override for custom behavior"""
        pass
    
    def set_parameter(self, name: str, value: Any):
        """Set an effect parameter"""
        if name in self.parameters:
            old_value = self.parameters[name]
            self.parameters[name] = value
            self.on_parameter_changed(name, old_value, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")
    
    def get_parameter(self, name: str) -> Any:
        """Get an effect parameter value"""
        return self.parameters.get(name, None)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all effect parameters"""
        return self.parameters.copy()
    
    def on_parameter_changed(self, name: str, old_value: Any, new_value: Any):
        """Called when a parameter is changed - override for custom behavior"""
        pass
    
    def reset(self):
        """Reset effect state"""
        self._initialize_state()
    
    def get_info(self) -> Dict[str, Any]:
        """Get effect information"""
        return {
            'name': self.__class__.__name__,
            'enabled': self.enabled,
            'parameters': self.parameters,
            'sample_rate': self.sample_rate
        }
    
    def apply_with_mix(self, audio_data: np.ndarray, mix: float = 1.0) -> np.ndarray:
        """
        Apply effect with wet/dry mix
        
        Args:
            audio_data: Input audio
            mix: Mix amount (0.0 = dry, 1.0 = wet)
            
        Returns:
            Mixed audio
        """
        if not self.enabled or mix <= 0.0:
            return audio_data
        
        # Process audio
        processed = self.process(audio_data)
        
        # Apply mix
        if mix >= 1.0:
            return processed
        else:
            return (1.0 - mix) * audio_data + mix * processed
    
    def apply_with_fade(self, audio_data: np.ndarray, fade_in: bool = False, fade_out: bool = False, fade_samples: int = 256) -> np.ndarray:
        """
        Apply effect with fade in/out to prevent clicks
        
        Args:
            audio_data: Input audio
            fade_in: Apply fade in
            fade_out: Apply fade out
            fade_samples: Number of samples for fade
            
        Returns:
            Faded audio
        """
        if not self.enabled:
            return audio_data
        
        processed = self.process(audio_data)
        
        if fade_in:
            fade_samples = min(fade_samples, len(processed))
            fade_curve = np.linspace(0, 1, fade_samples)
            processed[:fade_samples] *= fade_curve
            audio_data[:fade_samples] *= (1 - fade_curve)
            processed[:fade_samples] += audio_data[:fade_samples]
        
        if fade_out:
            fade_samples = min(fade_samples, len(processed))
            fade_curve = np.linspace(1, 0, fade_samples)
            processed[-fade_samples:] *= fade_curve
            audio_data[-fade_samples:] *= (1 - fade_curve)
            processed[-fade_samples:] += audio_data[-fade_samples:]
        
        return processed
