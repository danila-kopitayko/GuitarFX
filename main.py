#!/usr/bin/env python3
"""
Real-time Guitar Effects Processor with AI-driven Technique Detection
Main entry point for the application
"""

import sys
import tkinter as tk
from tkinter import messagebox
import threading
import time
import traceback

from audio.audio_manager import AudioManager
from audio.audio_processor import AudioProcessor
from detection.technique_detector import TechniqueDetector
from effects.effect_processor import EffectProcessor
from gui.main_window import MainWindow
from config.settings import Settings
from utils.logger import Logger

class GuitarEffectsApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        self.logger = Logger()
        self.settings = Settings()
        self.running = False
        
        # Initialize components
        self.audio_manager = None
        self.audio_processor = None
        self.technique_detector = None
        self.effect_processor = None
        self.gui = None
        
        # Threading
        self.audio_thread = None
        self.processing_thread = None
        
    def initialize(self):
        """Initialize all application components"""
        try:
            self.logger.info("Initializing Guitar Effects Processor...")
            
            # Initialize audio manager
            self.audio_manager = AudioManager(self.settings)
            
            # Initialize technique detector
            self.technique_detector = TechniqueDetector(self.settings)
            
            # Initialize effect processor
            self.effect_processor = EffectProcessor(self.settings)
            
            # Initialize audio processor
            self.audio_processor = AudioProcessor(
                self.audio_manager,
                self.technique_detector,
                self.effect_processor,
                self.settings
            )
            
            # Initialize GUI
            self.gui = MainWindow(
                self.audio_manager,
                self.audio_processor,
                self.technique_detector,
                self.effect_processor,
                self.settings
            )
            
            self.logger.info("Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def start_processing(self):
        """Start real-time audio processing"""
        if self.running:
            return
            
        try:
            self.running = True
            
            # Start audio stream
            self.audio_manager.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            self.logger.info("Audio processing started")
            
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self.running = False
            raise
    
    def stop_processing(self):
        """Stop real-time audio processing"""
        if not self.running:
            return
            
        self.running = False
        
        try:
            # Stop audio stream
            if self.audio_manager:
                self.audio_manager.stop_stream()
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            self.logger.info("Audio processing stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping processing: {e}")
    
    def _processing_loop(self):
        """Main audio processing loop"""
        while self.running:
            try:
                # Process one audio frame
                self.audio_processor.process_frame()
                
                # Small sleep to prevent CPU overload and allow GUI updates
                time.sleep(0.005)
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                # Continue processing despite errors
                time.sleep(0.05)
    
    def run(self):
        """Run the main application"""
        if not self.initialize():
            messagebox.showerror("Error", "Failed to initialize application")
            return
        
        try:
            # Setup GUI callbacks
            self.gui.set_start_callback(self.start_processing)
            self.gui.set_stop_callback(self.stop_processing)
            
            # Start GUI main loop
            self.gui.run()
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_processing()
            
            if self.audio_manager:
                self.audio_manager.cleanup()
                
            self.logger.info("Cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

def main():
    """Main entry point"""
    try:
        app = GuitarEffectsApp()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
