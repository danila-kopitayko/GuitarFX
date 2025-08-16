"""
Visualizer - Real-time audio visualization and analysis display
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
import time
from collections import deque

from utils.logger import Logger

class Visualizer:
    """Real-time audio visualization component"""
    
    def __init__(self, parent, audio_manager, audio_processor, settings):
        self.parent = parent
        self.audio_manager = audio_manager
        self.audio_processor = audio_processor
        self.settings = settings
        self.logger = Logger()
        
        # Visualization settings
        self.sample_rate = settings.get('audio.sample_rate', 44100)
        self.update_rate = settings.get('visualizer.update_rate', 30)  # Hz
        self.history_length = settings.get('visualizer.history_length', 200)  # frames
        
        # Data buffers
        self.audio_buffer = deque(maxlen=2048)
        self.spectrum_buffer = deque(maxlen=self.history_length)
        self.rms_history = deque(maxlen=self.history_length)
        self.technique_history = deque(maxlen=self.history_length)
        
        # Visualization state
        self.running = False
        self.last_update = 0
        
        # Create widgets
        self._create_widgets()
        
        # Setup plots
        self._setup_plots()
        
        # Start visualization thread
        self._start_visualization_thread()
    
    def _create_widgets(self):
        """Create visualization widgets"""
        # Configure parent
        self.parent.rowconfigure(0, weight=1)
        self.parent.columnconfigure(0, weight=1)
        
        # Create notebook for different visualization tabs
        self.notebook = ttk.Notebook(self.parent)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create visualization tabs
        self._create_waveform_tab()
        self._create_spectrum_tab()
        self._create_analysis_tab()
    
    def _create_waveform_tab(self):
        """Create waveform visualization tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Waveform")
        
        # Create matplotlib figure
        self.waveform_fig = Figure(figsize=(8, 4), dpi=100)
        self.waveform_ax = self.waveform_fig.add_subplot(111)
        
        # Canvas
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_fig, frame)
        self.waveform_canvas.draw()
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup waveform plot
        self.waveform_ax.set_title('Real-time Audio Waveform')
        self.waveform_ax.set_xlabel('Sample')
        self.waveform_ax.set_ylabel('Amplitude')
        self.waveform_ax.set_ylim(-1.1, 1.1)
        self.waveform_ax.grid(True, alpha=0.3)
        
        # Initialize empty line
        self.waveform_line, = self.waveform_ax.plot([], [], 'b-', linewidth=1)
    
    def _create_spectrum_tab(self):
        """Create spectrum analyzer tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Spectrum")
        
        # Create matplotlib figure
        self.spectrum_fig = Figure(figsize=(8, 4), dpi=100)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        
        # Canvas
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, frame)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup spectrum plot
        self.spectrum_ax.set_title('Real-time Spectrum Analyzer')
        self.spectrum_ax.set_xlabel('Frequency (Hz)')
        self.spectrum_ax.set_ylabel('Magnitude (dB)')
        self.spectrum_ax.set_xlim(20, 8000)
        self.spectrum_ax.set_ylim(-80, 0)
        self.spectrum_ax.set_xscale('log')
        self.spectrum_ax.grid(True, alpha=0.3)
        
        # Initialize empty line
        self.spectrum_line, = self.spectrum_ax.plot([], [], 'g-', linewidth=1)
        
        # Add frequency band markers
        self._add_frequency_markers()
    
    def _create_analysis_tab(self):
        """Create analysis and detection visualization tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Analysis")
        
        # Create matplotlib figure with subplots
        self.analysis_fig = Figure(figsize=(8, 6), dpi=100)
        
        # RMS level plot
        self.rms_ax = self.analysis_fig.add_subplot(311)
        self.rms_ax.set_title('Audio Level (RMS)')
        self.rms_ax.set_ylabel('Level (dB)')
        self.rms_ax.set_ylim(-60, 0)
        self.rms_ax.grid(True, alpha=0.3)
        self.rms_line, = self.rms_ax.plot([], [], 'r-', linewidth=1)
        
        # Technique detection plot
        self.technique_ax = self.analysis_fig.add_subplot(312)
        self.technique_ax.set_title('Technique Detection')
        self.technique_ax.set_ylabel('Confidence')
        self.technique_ax.set_ylim(0, 1)
        self.technique_ax.grid(True, alpha=0.3)
        self.technique_line, = self.technique_ax.plot([], [], 'purple', linewidth=2)
        
        # Spectrogram
        self.spectrogram_ax = self.analysis_fig.add_subplot(313)
        self.spectrogram_ax.set_title('Spectrogram')
        self.spectrogram_ax.set_xlabel('Time')
        self.spectrogram_ax.set_ylabel('Frequency (Hz)')
        self.spectrogram_ax.set_yscale('log')
        
        # Initialize spectrogram data
        self.spectrogram_data = np.zeros((50, self.history_length))
        self.spectrogram_image = self.spectrogram_ax.imshow(
            self.spectrogram_data,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        
        # Canvas
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, frame)
        self.analysis_canvas.draw()
        self.analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Adjust layout
        self.analysis_fig.tight_layout()
    
    def _add_frequency_markers(self):
        """Add frequency band markers to spectrum plot"""
        # Guitar frequency ranges
        markers = [
            (82, 'Low E'),
            (110, 'A'),
            (147, 'D'),
            (196, 'G'),
            (247, 'B'),
            (330, 'High E'),
            (1000, '1kHz'),
            (2000, '2kHz'),
            (4000, '4kHz')
        ]
        
        for freq, label in markers:
            self.spectrum_ax.axvline(freq, color='gray', alpha=0.5, linestyle='--')
            self.spectrum_ax.text(freq, -75, label, rotation=90, 
                                fontsize=8, alpha=0.7, ha='center')
    
    def _setup_plots(self):
        """Setup initial plot appearance"""
        # Set matplotlib style
        plt.style.use('default')
        
        # Configure figures
        for fig in [self.waveform_fig, self.spectrum_fig, self.analysis_fig]:
            fig.patch.set_facecolor('white')
    
    def _start_visualization_thread(self):
        """Start the visualization update thread"""
        self.running = True
        self.viz_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.viz_thread.start()
    
    def _visualization_loop(self):
        """Main visualization update loop"""
        while self.running:
            try:
                current_time = time.time()
                
                # Limit update rate
                if current_time - self.last_update < 1.0 / self.update_rate:
                    time.sleep(0.01)
                    continue
                
                # Get fresh audio data
                audio_frame = self.audio_manager.get_input_frame()
                if audio_frame is not None:
                    self._process_audio_data(audio_frame)
                
                # Update visualization
                self._update_plots()
                
                self.last_update = current_time
                
            except Exception as e:
                self.logger.error(f"Visualization update error: {e}")
                time.sleep(0.1)
    
    def _process_audio_data(self, audio_frame):
        """Process new audio data for visualization"""
        # Add to audio buffer
        self.audio_buffer.extend(audio_frame)
        
        # Compute RMS level
        rms = np.sqrt(np.mean(audio_frame ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))  # Convert to dB, avoid log(0)
        self.rms_history.append(rms_db)
        
        # Compute spectrum
        if len(audio_frame) > 64:  # Ensure we have enough samples
            spectrum = self._compute_spectrum(audio_frame)
            self.spectrum_buffer.append(spectrum)
        
        # Get technique detection info
        technique = self.audio_processor.get_current_technique()
        confidence = self.audio_processor.get_technique_confidence()
        
        # Store technique confidence
        if technique and technique != 'none':
            self.technique_history.append(confidence)
        else:
            self.technique_history.append(0.0)
    
    def _compute_spectrum(self, audio_data):
        """Compute frequency spectrum of audio data"""
        # Window the data
        windowed = audio_data * np.hanning(len(audio_data))
        
        # Compute FFT
        fft = np.fft.fft(windowed)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Convert to dB
        magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
        
        # Create frequency axis
        freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)[:len(fft)//2]
        
        return freqs, magnitude_db
    
    def _update_plots(self):
        """Update all visualization plots"""
        try:
            # Update waveform
            self._update_waveform()
            
            # Update spectrum
            self._update_spectrum()
            
            # Update analysis plots
            self._update_analysis()
            
        except Exception as e:
            self.logger.error(f"Plot update error: {e}")
    
    def _update_waveform(self):
        """Update waveform display"""
        if len(self.audio_buffer) > 0:
            # Get recent audio data
            waveform_data = list(self.audio_buffer)[-1024:]  # Last 1024 samples
            x_data = np.arange(len(waveform_data))
            
            # Update plot
            self.waveform_line.set_data(x_data, waveform_data)
            self.waveform_ax.set_xlim(0, len(waveform_data))
            
            # Update canvas
            self.waveform_canvas.draw_idle()
    
    def _update_spectrum(self):
        """Update spectrum analyzer display"""
        if len(self.spectrum_buffer) > 0:
            # Get latest spectrum
            freqs, magnitude_db = self.spectrum_buffer[-1]
            
            # Limit frequency range for display
            freq_mask = (freqs >= 20) & (freqs <= 8000)
            display_freqs = freqs[freq_mask]
            display_magnitude = magnitude_db[freq_mask]
            
            # Update plot
            self.spectrum_line.set_data(display_freqs, display_magnitude)
            
            # Update canvas
            self.spectrum_canvas.draw_idle()
    
    def _update_analysis(self):
        """Update analysis plots"""
        # Update RMS level plot
        if len(self.rms_history) > 0:
            x_data = np.arange(len(self.rms_history))
            self.rms_line.set_data(x_data, list(self.rms_history))
            self.rms_ax.set_xlim(0, len(self.rms_history))
            
            # Add current technique detection indicator
            current_technique = self.audio_processor.get_current_technique()
            if current_technique and current_technique != 'none':
                # Highlight current detection with background color
                self.rms_ax.axvspan(len(self.rms_history)-5, len(self.rms_history), 
                                  alpha=0.3, color='yellow')
        
        # Update technique confidence plot
        if len(self.technique_history) > 0:
            x_data = np.arange(len(self.technique_history))
            self.technique_line.set_data(x_data, list(self.technique_history))
            self.technique_ax.set_xlim(0, len(self.technique_history))
        
        # Update spectrogram
        if len(self.spectrum_buffer) > 10:  # Need some data for spectrogram
            self._update_spectrogram()
        
        # Update canvas
        self.analysis_canvas.draw_idle()
    
    def _update_spectrogram(self):
        """Update spectrogram display"""
        try:
            # Build spectrogram data from recent spectrum data
            recent_spectra = list(self.spectrum_buffer)[-self.history_length:]
            
            if len(recent_spectra) > 0:
                # Extract magnitude data and resample to fixed frequency bins
                freq_bins = 50  # Fixed number of frequency bins
                spectrogram_data = np.zeros((freq_bins, len(recent_spectra)))
                
                for i, (freqs, magnitude_db) in enumerate(recent_spectra):
                    # Resample magnitude data to fixed frequency bins
                    target_freqs = np.logspace(np.log10(20), np.log10(8000), freq_bins)
                    
                    # Interpolate magnitude values
                    if len(freqs) > 0 and len(magnitude_db) > 0:
                        interp_magnitude = np.interp(target_freqs, freqs, magnitude_db)
                        spectrogram_data[:, i] = interp_magnitude
                
                # Update spectrogram image
                self.spectrogram_image.set_array(spectrogram_data)
                self.spectrogram_image.set_clim(vmin=-80, vmax=0)
                
                # Update frequency labels
                freq_ticks = [20, 100, 500, 2000, 8000]
                freq_tick_positions = [np.interp(f, np.logspace(np.log10(20), np.log10(8000), freq_bins), 
                                                np.arange(freq_bins)) for f in freq_ticks]
                self.spectrogram_ax.set_yticks(freq_tick_positions)
                self.spectrogram_ax.set_yticklabels(freq_ticks)
                
        except Exception as e:
            self.logger.error(f"Spectrogram update error: {e}")
    
    def update(self):
        """External update call from main GUI"""
        # This is called from the main GUI thread
        # Actual visualization updates happen in the visualization thread
        pass
    
    def stop(self):
        """Stop visualization"""
        self.running = False
        if hasattr(self, 'viz_thread'):
            self.viz_thread.join(timeout=1.0)
    
    def get_status(self):
        """Get visualizer status"""
        return {
            'running': self.running,
            'update_rate': self.update_rate,
            'audio_buffer_size': len(self.audio_buffer),
            'spectrum_buffer_size': len(self.spectrum_buffer),
            'rms_history_size': len(self.rms_history),
            'technique_history_size': len(self.technique_history)
        }
