"""
Main Window - Primary GUI interface for the guitar effects processor
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

from gui.control_panel import ControlPanel
from gui.visualizer import Visualizer
from utils.logger import Logger

class MainWindow:
    """Main application window with all GUI components"""
    
    def __init__(self, audio_manager, audio_processor, technique_detector, effect_processor, settings):
        self.audio_manager = audio_manager
        self.audio_processor = audio_processor
        self.technique_detector = technique_detector
        self.effect_processor = effect_processor
        self.settings = settings
        self.logger = Logger()
        
        # GUI state
        self.running = False
        self.update_thread = None
        
        # Callbacks
        self.start_callback = None
        self.stop_callback = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Guitar Effects Processor - AI Technique Detection")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Configure style
        self._setup_styles()
        
        # Create GUI components
        self._create_widgets()
        
        # Setup window events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start GUI update thread
        self._start_update_thread()
    
    def _setup_styles(self):
        """Setup GUI styles and themes"""
        style = ttk.Style()
        
        # Configure theme
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass  # Fall back to default theme
        
        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        
        # Button styles
        style.configure('Start.TButton', font=('Arial', 12, 'bold'))
        style.configure('Stop.TButton', font=('Arial', 12, 'bold'))
        
        # Configure colors
        self.root.configure(bg='#f0f0f0')
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Real-time Guitar Effects Processor",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel - Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Audio Analysis", padding="10")
        viz_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        # Create control panel
        self.control_panel = ControlPanel(
            control_frame,
            self.audio_manager,
            self.audio_processor,
            self.technique_detector,
            self.effect_processor,
            self.settings
        )
        
        # Create visualizer
        self.visualizer = Visualizer(
            viz_frame,
            self.audio_manager,
            self.audio_processor,
            self.settings
        )
        
        # Create status bar
        self._create_status_bar(status_frame)
        
        # Create menu bar
        self._create_menu_bar()
    
    def _create_status_bar(self, parent):
        """Create status bar with system information"""
        # Status label
        self.status_label = ttk.Label(
            parent,
            text="Ready",
            style='Status.TLabel'
        )
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Performance info
        self.perf_label = ttk.Label(
            parent,
            text="",
            style='Status.TLabel'
        )
        self.perf_label.grid(row=0, column=1, sticky=tk.E)
        
        # Technique status
        self.technique_label = ttk.Label(
            parent,
            text="Technique: None",
            style='Status.TLabel'
        )
        self.technique_label.grid(row=0, column=2, sticky=tk.E, padx=(20, 0))
    
    def _create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reset Settings", command=self._reset_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Audio menu
        audio_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Audio", menu=audio_menu)
        audio_menu.add_command(label="Refresh Devices", command=self._refresh_devices)
        audio_menu.add_command(label="Audio Test", command=self._audio_test)
        
        # Effects menu
        effects_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Effects", menu=effects_menu)
        effects_menu.add_command(label="Reset All Effects", command=self._reset_effects)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _start_update_thread(self):
        """Start the GUI update thread"""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Main GUI update loop"""
        while True:
            try:
                if self.root.winfo_exists():
                    self.root.after(0, self._update_gui)
                    time.sleep(0.1)  # Update at 10 Hz
                else:
                    break
            except tk.TclError:
                break
            except Exception as e:
                self.logger.error(f"GUI update error: {e}")
                time.sleep(1)
    
    def _update_gui(self):
        """Update GUI elements"""
        try:
            # Update status
            self._update_status()
            
            # Update control panel
            self.control_panel.update()
            
            # Update visualizer
            self.visualizer.update()
            
        except Exception as e:
            self.logger.error(f"GUI update error: {e}")
    
    def _update_status(self):
        """Update status bar information"""
        try:
            # Audio processing status
            if self.audio_manager.is_stream_active():
                self.status_label.config(text="Processing Active")
            else:
                self.status_label.config(text="Stopped")
            
            # Performance information
            stats = self.audio_processor.get_performance_stats()
            frame_count = stats.get('frame_count', 0)
            self.perf_label.config(text=f"Frames: {frame_count}")
            
            # Current technique
            technique = self.audio_processor.get_current_technique()
            confidence = self.audio_processor.get_technique_confidence()
            
            if technique and technique != 'none':
                self.technique_label.config(
                    text=f"Technique: {technique.title()} ({confidence:.1%})"
                )
            else:
                self.technique_label.config(text="Technique: None")
                
        except Exception as e:
            self.logger.error(f"Status update error: {e}")
    
    def set_start_callback(self, callback):
        """Set callback for start button"""
        self.start_callback = callback
        self.control_panel.set_start_callback(callback)
    
    def set_stop_callback(self, callback):
        """Set callback for stop button"""
        self.stop_callback = callback
        self.control_panel.set_stop_callback(callback)
    
    def _refresh_devices(self):
        """Refresh audio devices"""
        try:
            self.audio_manager._scan_devices()
            self.control_panel.refresh_device_lists()
            messagebox.showinfo("Success", "Audio devices refreshed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh devices: {e}")
    
    def _audio_test(self):
        """Run audio system test"""
        try:
            stream_info = self.audio_manager.get_stream_info()
            info_text = f"""Audio System Information:
            
Sample Rate: {stream_info['sample_rate']} Hz
Buffer Size: {stream_info['buffer_size']} samples
Channels: {stream_info['channels']}
Format: {stream_info['format']}
Stream Active: {stream_info['active']}
Input Latency: {stream_info['input_latency']:.1f} ms
Output Latency: {stream_info['output_latency']:.1f} ms"""
            
            messagebox.showinfo("Audio Test", info_text)
            
        except Exception as e:
            messagebox.showerror("Audio Test", f"Audio test failed: {e}")
    
    def _reset_effects(self):
        """Reset all effects to default settings"""
        try:
            self.effect_processor.clear_all_effects()
            self.control_panel.reset_effect_controls()
            messagebox.showinfo("Success", "All effects reset")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset effects: {e}")
    
    def _reset_settings(self):
        """Reset application settings"""
        result = messagebox.askyesno(
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?"
        )
        if result:
            try:
                self.settings.reset_to_defaults()
                messagebox.showinfo("Success", "Settings reset. Please restart the application.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset settings: {e}")
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """Guitar Effects Processor v1.0

Real-time AI-driven technique detection and effects processing for guitar.

Features:
• Real-time audio processing
• AI technique detection (chugging, harmonics)
• Cyberpunk distortion effect
• Broken record vinyl effect
• Low-latency audio pipeline
• Expandable architecture

Built with Python, PyAudio, scikit-learn, and tkinter."""
        
        messagebox.showinfo("About", about_text)
    
    def _on_closing(self):
        """Handle window closing"""
        if self.stop_callback:
            self.stop_callback()
        
        # Give time for cleanup
        time.sleep(0.5)
        
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def run(self):
        """Run the GUI main loop"""
        try:
            self.logger.info("Starting GUI")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"GUI error: {e}")
        finally:
            self.logger.info("GUI shutdown")
