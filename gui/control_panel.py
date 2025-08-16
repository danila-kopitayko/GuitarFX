"""
Control Panel - GUI controls for audio settings and effects
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, Dict, Any

from utils.logger import Logger

class ControlPanel:
    """Control panel with audio settings and effect controls"""
    
    def __init__(self, parent, audio_manager, audio_processor, technique_detector, effect_processor, settings):
        self.parent = parent
        self.audio_manager = audio_manager
        self.audio_processor = audio_processor
        self.technique_detector = technique_detector
        self.effect_processor = effect_processor
        self.settings = settings
        self.logger = Logger()
        
        # Callbacks
        self.start_callback = None
        self.stop_callback = None
        
        # GUI variables
        self.input_device_var = tk.StringVar()
        self.output_device_var = tk.StringVar()
        self.processing_enabled_var = tk.BooleanVar()
        self.bypass_effects_var = tk.BooleanVar()
        
        # Effect parameter variables
        self.effect_vars = {}
        
        # Create widgets
        self._create_widgets()
        
        # Initialize values
        self._initialize_values()
        
        # Bind events
        self._bind_events()
    
    def _create_widgets(self):
        """Create all control panel widgets"""
        # Configure parent grid
        self.parent.rowconfigure(0, weight=0)  # Controls
        self.parent.rowconfigure(1, weight=1)  # Effects
        
        # Main controls section
        controls_frame = ttk.LabelFrame(self.parent, text="Audio Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)
        
        self._create_audio_controls(controls_frame)
        
        # Effects section
        effects_frame = ttk.LabelFrame(self.parent, text="Effects", padding="10")
        effects_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        effects_frame.columnconfigure(0, weight=1)
        
        self._create_effects_controls(effects_frame)
    
    def _create_audio_controls(self, parent):
        """Create audio device and processing controls"""
        row = 0
        
        # Start/Stop buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=2, pady=(0, 10))
        
        self.start_button = ttk.Button(
            button_frame,
            text="Start Processing",
            style='Start.TButton',
            command=self._on_start_clicked
        )
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Processing",
            style='Stop.TButton',
            command=self._on_stop_clicked,
            state='disabled'
        )
        self.stop_button.grid(row=0, column=1, padx=(5, 0))
        
        row += 1
        
        # Processing enable checkbox
        processing_check = ttk.Checkbutton(
            parent,
            text="Enable Processing",
            variable=self.processing_enabled_var,
            command=self._on_processing_toggle
        )
        processing_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Bypass effects checkbox
        bypass_check = ttk.Checkbutton(
            parent,
            text="Bypass Effects",
            variable=self.bypass_effects_var,
            command=self._on_bypass_toggle
        )
        bypass_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Input device selection
        ttk.Label(parent, text="Input Device:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.input_combo = ttk.Combobox(
            parent,
            textvariable=self.input_device_var,
            state='readonly'
        )
        self.input_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.input_combo.bind('<<ComboboxSelected>>', self._on_input_device_changed)
        row += 1
        
        # Output device selection
        ttk.Label(parent, text="Output Device:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.output_combo = ttk.Combobox(
            parent,
            textvariable=self.output_device_var,
            state='readonly'
        )
        self.output_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.output_combo.bind('<<ComboboxSelected>>', self._on_output_device_changed)
        row += 1
        
        # Audio settings
        ttk.Label(parent, text="Sample Rate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.sample_rate_var = tk.StringVar(value="44100")
        sample_rate_combo = ttk.Combobox(
            parent,
            textvariable=self.sample_rate_var,
            values=["22050", "44100", "48000", "96000"],
            state='readonly',
            width=10
        )
        sample_rate_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        sample_rate_combo.bind('<<ComboboxSelected>>', self._on_sample_rate_changed)
        row += 1
        
        ttk.Label(parent, text="Buffer Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.buffer_size_var = tk.StringVar(value="512")
        buffer_size_combo = ttk.Combobox(
            parent,
            textvariable=self.buffer_size_var,
            values=["128", "256", "512", "1024", "2048"],
            state='readonly',
            width=10
        )
        buffer_size_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        buffer_size_combo.bind('<<ComboboxSelected>>', self._on_buffer_size_changed)
        row += 1
    
    def _create_effects_controls(self, parent):
        """Create effects control widgets"""
        # Create notebook for effect tabs
        self.effects_notebook = ttk.Notebook(parent)
        self.effects_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.rowconfigure(0, weight=1)
        
        # Create effect control tabs
        self._create_distortion_tab()
        self._create_broken_record_tab()
        self._create_detection_tab()
    
    def _create_distortion_tab(self):
        """Create distortion effect controls"""
        frame = ttk.Frame(self.effects_notebook)
        self.effects_notebook.add(frame, text="Distortion")
        
        # Distortion effect parameters
        distortion_effect = self.effect_processor.get_effect('distortion')
        if distortion_effect:
            params = distortion_effect.get_parameters()
            
            row = 0
            for param_name, value in params.items():
                self._create_parameter_control(
                    frame, 'distortion', param_name, value, row
                )
                row += 1
            
            # Preset selection
            row += 1
            ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            row += 1
            
            ttk.Label(frame, text="Preset:").grid(row=row, column=0, sticky=tk.W, pady=5)
            preset_var = tk.StringVar()
            preset_combo = ttk.Combobox(
                frame,
                textvariable=preset_var,
                values=["light", "medium", "heavy", "cyberpunk"],
                state='readonly'
            )
            preset_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
            preset_combo.bind('<<ComboboxSelected>>', 
                lambda e: self._load_effect_preset('distortion', preset_var.get()))
    
    def _create_broken_record_tab(self):
        """Create broken record effect controls"""
        frame = ttk.Frame(self.effects_notebook)
        self.effects_notebook.add(frame, text="Broken Record")
        
        # Broken record effect parameters
        broken_record_effect = self.effect_processor.get_effect('broken_record')
        if broken_record_effect:
            params = broken_record_effect.get_parameters()
            
            row = 0
            for param_name, value in params.items():
                self._create_parameter_control(
                    frame, 'broken_record', param_name, value, row
                )
                row += 1
            
            # Preset selection
            row += 1
            ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            row += 1
            
            ttk.Label(frame, text="Preset:").grid(row=row, column=0, sticky=tk.W, pady=5)
            preset_var = tk.StringVar()
            preset_combo = ttk.Combobox(
                frame,
                textvariable=preset_var,
                values=["light_wear", "medium_wear", "heavy_damage", "broken"],
                state='readonly'
            )
            preset_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
            preset_combo.bind('<<ComboboxSelected>>', 
                lambda e: self._load_effect_preset('broken_record', preset_var.get()))
    
    def _create_detection_tab(self):
        """Create technique detection controls"""
        frame = ttk.Frame(self.effects_notebook)
        self.effects_notebook.add(frame, text="Detection")
        
        row = 0
        
        # Detection settings
        ttk.Label(frame, text="Confidence Threshold:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.7)
        confidence_scale = ttk.Scale(
            frame,
            from_=0.1,
            to=0.95,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
            command=self._on_confidence_changed
        )
        confidence_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        
        self.confidence_label = ttk.Label(frame, text="0.70")
        self.confidence_label.grid(row=row, column=2, sticky=tk.W, padx=(5, 0), pady=5)
        row += 1
        
        # Technique mapping
        ttk.Label(frame, text="Technique Mapping:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(15, 5)
        )
        row += 1
        
        # Chugging -> Distortion
        ttk.Label(frame, text="Chugging →").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(frame, text="Distortion Effect").grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Harmonic -> Broken Record
        ttk.Label(frame, text="Pinch Harmonic →").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(frame, text="Broken Record Effect").grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Detection status
        row += 1
        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        ttk.Label(frame, text="Detection Status:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        row += 1
        
        self.detection_status_label = ttk.Label(frame, text="No technique detected")
        self.detection_status_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1
        
        # Configure column weights
        frame.columnconfigure(1, weight=1)
    
    def _create_parameter_control(self, parent, effect_name: str, param_name: str, value: Any, row: int):
        """Create a parameter control widget"""
        # Parameter label
        display_name = param_name.replace('_', ' ').title()
        ttk.Label(parent, text=f"{display_name}:").grid(row=row, column=0, sticky=tk.W, pady=2)
        
        # Create appropriate control based on parameter type and range
        if isinstance(value, bool):
            # Boolean parameter - checkbox
            var = tk.BooleanVar(value=value)
            control = ttk.Checkbutton(
                parent,
                variable=var,
                command=lambda: self._on_parameter_changed(effect_name, param_name, var.get())
            )
            control.grid(row=row, column=1, sticky=tk.W, pady=2)
        
        elif isinstance(value, (int, float)):
            # Numeric parameter - scale
            if param_name in ['low_cut', 'high_cut']:
                # Frequency parameters
                if 'low' in param_name:
                    min_val, max_val = 20, 500
                else:
                    min_val, max_val = 1000, 20000
            elif param_name == 'sample_rate':
                min_val, max_val = 22050, 96000
            else:
                # General 0-1 range parameters
                min_val, max_val = 0.0, 1.0
            
            var = tk.DoubleVar(value=value)
            control = ttk.Scale(
                parent,
                from_=min_val,
                to=max_val,
                variable=var,
                orient=tk.HORIZONTAL,
                command=lambda v: self._on_parameter_changed(effect_name, param_name, var.get())
            )
            control.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            
            # Value label
            value_label = ttk.Label(parent, text=f"{value:.2f}")
            value_label.grid(row=row, column=2, sticky=tk.W, padx=(5, 0), pady=2)
            
            # Store reference for updates
            var.trace('w', lambda *args: value_label.config(text=f"{var.get():.2f}"))
        
        else:
            # String or other - entry
            var = tk.StringVar(value=str(value))
            control = ttk.Entry(
                parent,
                textvariable=var
            )
            control.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            control.bind('<Return>', lambda e: self._on_parameter_changed(effect_name, param_name, var.get()))
        
        # Store variable reference
        if effect_name not in self.effect_vars:
            self.effect_vars[effect_name] = {}
        self.effect_vars[effect_name][param_name] = var
        
        # Configure column weight
        parent.columnconfigure(1, weight=1)
    
    def _initialize_values(self):
        """Initialize control values from current settings"""
        try:
            # Initialize device lists
            self.refresh_device_lists()
            
            # Set current devices
            input_device = self.settings.get('audio.input_device')
            output_device = self.settings.get('audio.output_device')
            
            if input_device is not None and input_device < len(self.input_combo['values']):
                self.input_combo.current(input_device)
            
            if output_device is not None and output_device < len(self.output_combo['values']):
                self.output_combo.current(output_device)
            
            # Set audio settings
            self.sample_rate_var.set(str(self.settings.get('audio.sample_rate', 44100)))
            self.buffer_size_var.set(str(self.settings.get('audio.buffer_size', 512)))
            
            # Set processing state
            self.processing_enabled_var.set(True)
            self.bypass_effects_var.set(False)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize values: {e}")
    
    def _bind_events(self):
        """Bind GUI events"""
        # Enable processing by default
        self.processing_enabled_var.set(True)
        self._on_processing_toggle()
    
    def refresh_device_lists(self):
        """Refresh audio device lists"""
        try:
            # Get input devices
            input_devices = self.audio_manager.get_input_devices()
            input_names = [f"{i}: {dev['name']}" for i, dev in enumerate(input_devices)]
            self.input_combo['values'] = input_names
            
            # Get output devices
            output_devices = self.audio_manager.get_output_devices()
            output_names = [f"{i}: {dev['name']}" for i, dev in enumerate(output_devices)]
            self.output_combo['values'] = output_names
            
            # Select default devices if none selected
            if not self.input_combo.get() and input_names:
                self.input_combo.current(0)
            
            if not self.output_combo.get() and output_names:
                self.output_combo.current(0)
                
        except Exception as e:
            self.logger.error(f"Failed to refresh device lists: {e}")
    
    def _on_start_clicked(self):
        """Handle start button click"""
        try:
            if self.start_callback:
                self.start_callback()
                
            # Update button states
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            
        except Exception as e:
            self.logger.error(f"Start failed: {e}")
            messagebox.showerror("Error", f"Failed to start processing: {e}")
    
    def _on_stop_clicked(self):
        """Handle stop button click"""
        try:
            if self.stop_callback:
                self.stop_callback()
                
            # Update button states
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
        except Exception as e:
            self.logger.error(f"Stop failed: {e}")
    
    def _on_processing_toggle(self):
        """Handle processing enable/disable toggle"""
        enabled = self.processing_enabled_var.get()
        if enabled:
            self.audio_processor.enable_processing()
        else:
            self.audio_processor.disable_processing()
    
    def _on_bypass_toggle(self):
        """Handle effects bypass toggle"""
        bypass = self.bypass_effects_var.get()
        self.audio_processor.set_bypass(bypass)
    
    def _on_input_device_changed(self, event=None):
        """Handle input device selection change"""
        try:
            selection = self.input_combo.current()
            if selection >= 0:
                self.audio_manager.set_input_device(selection)
                self.settings.set('audio.input_device', selection)
        except Exception as e:
            self.logger.error(f"Failed to set input device: {e}")
    
    def _on_output_device_changed(self, event=None):
        """Handle output device selection change"""
        try:
            selection = self.output_combo.current()
            if selection >= 0:
                self.audio_manager.set_output_device(selection)
                self.settings.set('audio.output_device', selection)
        except Exception as e:
            self.logger.error(f"Failed to set output device: {e}")
    
    def _on_sample_rate_changed(self, event=None):
        """Handle sample rate change"""
        try:
            sample_rate = int(self.sample_rate_var.get())
            self.settings.set('audio.sample_rate', sample_rate)
        except Exception as e:
            self.logger.error(f"Failed to set sample rate: {e}")
    
    def _on_buffer_size_changed(self, event=None):
        """Handle buffer size change"""
        try:
            buffer_size = int(self.buffer_size_var.get())
            self.settings.set('audio.buffer_size', buffer_size)
        except Exception as e:
            self.logger.error(f"Failed to set buffer size: {e}")
    
    def _on_parameter_changed(self, effect_name: str, param_name: str, value: Any):
        """Handle effect parameter change"""
        try:
            self.effect_processor.set_effect_parameter(effect_name, param_name, value)
            self.logger.debug(f"Set {effect_name}.{param_name} = {value}")
        except Exception as e:
            self.logger.error(f"Failed to set parameter {effect_name}.{param_name}: {e}")
    
    def _on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        confidence = float(value)
        self.confidence_label.config(text=f"{confidence:.2f}")
        self.settings.set('detection.min_confidence', confidence)
    
    def _load_effect_preset(self, effect_name: str, preset_name: str):
        """Load preset for an effect"""
        if self.effect_processor.load_effect_preset(effect_name, preset_name):
            # Update GUI controls to reflect new values
            self._update_effect_controls(effect_name)
    
    def _update_effect_controls(self, effect_name: str):
        """Update GUI controls for an effect"""
        try:
            effect = self.effect_processor.get_effect(effect_name)
            if effect and effect_name in self.effect_vars:
                params = effect.get_parameters()
                for param_name, value in params.items():
                    if param_name in self.effect_vars[effect_name]:
                        var = self.effect_vars[effect_name][param_name]
                        if isinstance(var, tk.BooleanVar):
                            var.set(bool(value))
                        elif isinstance(var, (tk.DoubleVar, tk.IntVar)):
                            var.set(float(value))
                        else:
                            var.set(str(value))
        except Exception as e:
            self.logger.error(f"Failed to update effect controls: {e}")
    
    def set_start_callback(self, callback: Callable):
        """Set start button callback"""
        self.start_callback = callback
    
    def set_stop_callback(self, callback: Callable):
        """Set stop button callback"""
        self.stop_callback = callback
    
    def reset_effect_controls(self):
        """Reset all effect controls to default values"""
        try:
            for effect_name in self.effect_vars:
                self._update_effect_controls(effect_name)
        except Exception as e:
            self.logger.error(f"Failed to reset effect controls: {e}")
    
    def update(self):
        """Update control panel state"""
        try:
            # Update detection status
            technique = self.audio_processor.get_current_technique()
            confidence = self.audio_processor.get_technique_confidence()
            
            if technique and technique != 'none':
                status_text = f"Detected: {technique.title()} (Confidence: {confidence:.1%})"
            else:
                status_text = "No technique detected"
            
            self.detection_status_label.config(text=status_text)
            
            # Update processing status indicators
            if self.audio_manager.is_stream_active():
                if not self.processing_enabled_var.get():
                    self.processing_enabled_var.set(True)
            
        except Exception as e:
            self.logger.error(f"Control panel update error: {e}")
