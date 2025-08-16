"""
Effect Processor - Manages and coordinates all effects
"""

import numpy as np
from typing import Dict, Optional, List
import time

from effects.distortion_effect import DistortionEffect
from effects.basic_distortion_effect import BasicDistortionEffect
from effects.broken_record_effect import BrokenRecordEffect
from utils.logger import Logger

class EffectProcessor:
    """Manages and processes all audio effects"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger()
        
        # Audio parameters
        self.sample_rate = settings.get('audio.sample_rate', 44100)
        
        # Effects registry
        self.effects = {}
        self.effect_chain = []
        
        # Current technique and effect mapping
        self.current_technique = None
        self.technique_confidence = 0.0
        self.active_effects = set()
        
        # Effect mapping configuration
        self.technique_effects = {
            'chugging': ['distortion'],
            'harmonic': ['broken_record'],
            'none': ['basic_distortion']
        }
        
        # Crossfade parameters
        self.crossfade_enabled = settings.get('effects.crossfade_enabled', True)
        self.crossfade_time = settings.get('effects.crossfade_time', 0.05)  # 50ms
        self.crossfade_samples = int(self.crossfade_time * self.sample_rate)
        
        # Effect states for crossfading
        self.effect_states = {}
        self.fade_counters = {}
        
        # Initialize effects
        self._initialize_effects()
    
    def _initialize_effects(self):
        """Initialize all available effects"""
        try:
            # Create distortion effect (cyberpunk-style for chugging)
            self.effects['distortion'] = DistortionEffect(
                sample_rate=self.sample_rate,
                **self.settings.get('effects.distortion', {})
            )
            
            # Create basic distortion effect (gentle for normal playing)
            self.effects['basic_distortion'] = BasicDistortionEffect(
                sample_rate=self.sample_rate,
                **self.settings.get('effects.basic_distortion', {})
            )
            
            # Create broken record effect
            self.effects['broken_record'] = BrokenRecordEffect(
                sample_rate=self.sample_rate,
                **self.settings.get('effects.broken_record', {})
            )
            
            # Initialize effect states
            for effect_name in self.effects:
                self.effect_states[effect_name] = 'off'  # 'off', 'fading_in', 'on', 'fading_out'
                self.fade_counters[effect_name] = 0
            
            self.logger.info(f"Initialized {len(self.effects)} effects")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize effects: {e}")
    
    def set_technique(self, technique: Optional[str], confidence: float):
        """Set the current playing technique and activate appropriate effects"""
        if technique == self.current_technique:
            self.technique_confidence = confidence
            return
        
        self.logger.debug(f"Technique change: {self.current_technique} -> {technique} (confidence: {confidence:.2f})")
        
        old_technique = self.current_technique
        self.current_technique = technique
        self.technique_confidence = confidence
        
        # Determine which effects should be active
        if technique and technique in self.technique_effects:
            target_effects = set(self.technique_effects[technique])
        else:
            target_effects = set()
        
        # Handle effect transitions
        self._transition_effects(target_effects, confidence)
    
    def _transition_effects(self, target_effects: set, confidence: float):
        """Transition effects based on target state"""
        current_active = set(effect for effect in self.active_effects)
        
        # Effects to turn on
        effects_to_enable = target_effects - current_active
        
        # Effects to turn off
        effects_to_disable = current_active - target_effects
        
        # Handle effect enabling
        for effect_name in effects_to_enable:
            if effect_name in self.effects:
                self._start_effect_fade_in(effect_name, confidence)
        
        # Handle effect disabling
        for effect_name in effects_to_disable:
            if effect_name in self.effects:
                self._start_effect_fade_out(effect_name)
    
    def _start_effect_fade_in(self, effect_name: str, confidence: float):
        """Start fading in an effect"""
        if self.crossfade_enabled:
            self.effect_states[effect_name] = 'fading_in'
            self.fade_counters[effect_name] = 0
        else:
            self.effect_states[effect_name] = 'on'
            self.effects[effect_name].enable()
        
        self.active_effects.add(effect_name)
        self.logger.debug(f"Starting fade-in for effect: {effect_name}")
    
    def _start_effect_fade_out(self, effect_name: str):
        """Start fading out an effect"""
        if self.crossfade_enabled:
            self.effect_states[effect_name] = 'fading_out'
            self.fade_counters[effect_name] = 0
        else:
            self.effect_states[effect_name] = 'off'
            self.effects[effect_name].disable()
            self.active_effects.discard(effect_name)
        
        self.logger.debug(f"Starting fade-out for effect: {effect_name}")
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio through active effects with crossfading"""
        if len(audio_data) == 0:
            return audio_data
        
        # Start with dry signal
        processed = audio_data.copy()
        
        # Process each potentially active effect
        for effect_name, effect in self.effects.items():
            state = self.effect_states.get(effect_name, 'off')
            
            if state == 'off':
                continue
            
            # Process audio through effect
            try:
                effect_output = effect.process(audio_data)
            except Exception as e:
                self.logger.error(f"Effect {effect_name} processing error: {e}")
                continue
            
            # Apply crossfading
            if state == 'on':
                # Effect fully on
                processed = effect_output
                
            elif state == 'fading_in':
                # Fade in effect
                fade_progress = self.fade_counters[effect_name] / self.crossfade_samples
                fade_progress = min(fade_progress, 1.0)
                
                # Apply fade curve (smooth)
                fade_curve = self._get_fade_curve(fade_progress)
                processed = processed * (1 - fade_curve) + effect_output * fade_curve
                
                # Update fade counter
                self.fade_counters[effect_name] += len(audio_data)
                
                # Check if fade complete
                if fade_progress >= 1.0:
                    self.effect_states[effect_name] = 'on'
                    effect.enable()
                    self.logger.debug(f"Fade-in complete for effect: {effect_name}")
                
            elif state == 'fading_out':
                # Fade out effect
                fade_progress = self.fade_counters[effect_name] / self.crossfade_samples
                fade_progress = min(fade_progress, 1.0)
                
                # Apply fade curve (smooth)
                fade_curve = self._get_fade_curve(1.0 - fade_progress)
                processed = processed * (1 - fade_curve) + effect_output * fade_curve
                
                # Update fade counter
                self.fade_counters[effect_name] += len(audio_data)
                
                # Check if fade complete
                if fade_progress >= 1.0:
                    self.effect_states[effect_name] = 'off'
                    effect.disable()
                    self.active_effects.discard(effect_name)
                    self.logger.debug(f"Fade-out complete for effect: {effect_name}")
        
        return processed
    
    def _get_fade_curve(self, progress: float) -> float:
        """Get smooth fade curve value"""
        # Use cosine curve for smooth fading
        return 0.5 * (1.0 - np.cos(progress * np.pi))
    
    def clear_all_effects(self):
        """Clear all active effects immediately"""
        for effect_name in list(self.active_effects):
            self.effect_states[effect_name] = 'off'
            self.effects[effect_name].disable()
            self.fade_counters[effect_name] = 0
        
        self.active_effects.clear()
        self.current_technique = None
        self.technique_confidence = 0.0
        
        self.logger.debug("All effects cleared")
    
    def get_effect(self, effect_name: str):
        """Get effect instance by name"""
        return self.effects.get(effect_name)
    
    def set_effect_parameter(self, effect_name: str, parameter: str, value):
        """Set parameter for a specific effect"""
        if effect_name in self.effects:
            try:
                self.effects[effect_name].set_parameter(parameter, value)
                return True
            except Exception as e:
                self.logger.error(f"Failed to set {effect_name}.{parameter} = {value}: {e}")
        return False
    
    def get_effect_parameter(self, effect_name: str, parameter: str):
        """Get parameter value from a specific effect"""
        if effect_name in self.effects:
            return self.effects[effect_name].get_parameter(parameter)
        return None
    
    def load_effect_preset(self, effect_name: str, preset_name: str):
        """Load a preset for a specific effect"""
        if effect_name in self.effects:
            try:
                self.effects[effect_name].load_preset(preset_name)
                self.logger.info(f"Loaded preset '{preset_name}' for effect '{effect_name}'")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load preset '{preset_name}' for effect '{effect_name}': {e}")
        return False
    
    def get_available_effects(self) -> List[str]:
        """Get list of available effects"""
        return list(self.effects.keys())
    
    def get_active_effects(self) -> List[str]:
        """Get list of currently active effects"""
        return list(self.active_effects)
    
    def get_effect_states(self) -> Dict[str, str]:
        """Get current state of all effects"""
        return self.effect_states.copy()
    
    def get_status(self) -> Dict:
        """Get current processor status"""
        return {
            'current_technique': self.current_technique,
            'technique_confidence': self.technique_confidence,
            'active_effects': list(self.active_effects),
            'effect_states': self.effect_states.copy(),
            'available_effects': list(self.effects.keys()),
            'crossfade_enabled': self.crossfade_enabled,
            'crossfade_time': self.crossfade_time
        }
    
    def set_technique_mapping(self, technique: str, effects: List[str]):
        """Set which effects are triggered by a technique"""
        if all(effect in self.effects for effect in effects):
            self.technique_effects[technique] = effects
            self.logger.info(f"Set technique mapping: {technique} -> {effects}")
            return True
        return False
    
    def get_technique_mapping(self) -> Dict[str, List[str]]:
        """Get current technique to effect mappings"""
        return self.technique_effects.copy()
