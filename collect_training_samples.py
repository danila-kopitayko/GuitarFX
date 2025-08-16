#!/usr/bin/env python3
"""
Training Sample Collection Helper
Record guitar techniques for training the AI model
"""

import argparse
import pyaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import time
import threading
import queue
import keyboard

from config.settings import Settings
from utils.logger import Logger


class SampleRecorder:
    """Records guitar technique samples for training"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = Logger()
        
        # Audio settings
        self.sample_rate = self.settings.get('audio.sample_rate', 44100)
        self.buffer_size = self.settings.get('audio.buffer_size', 1024)
        self.channels = 1
        self.format = pyaudio.paFloat32
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recorded_data = []
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
    def list_audio_devices(self):
        """List available audio input devices"""
        print("Available audio input devices:")
        print("-" * 50)
        
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"{i}: {device_info['name']}")
                print(f"   Channels: {device_info['maxInputChannels']}")
                print(f"   Sample Rate: {device_info['defaultSampleRate']}")
                print()
    
    def start_recording_session(self, output_dir: str, technique: str, device_id: int = None):
        """Start interactive recording session"""
        output_path = Path(output_dir) / technique
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get existing sample count
        existing_files = list(output_path.glob(f"{technique}_sample_*.wav"))
        sample_count = len(existing_files) + 1
        
        print(f"\n=== Recording {technique.upper()} samples ===")
        print(f"Output directory: {output_path}")
        print(f"Starting at sample #{sample_count}")
        print("\nControls:")
        print("  SPACE: Start/Stop recording")
        print("  ESC: Exit")
        print("  'd': Delete last recording")
        print("\nPress SPACE to start your first recording...")
        
        try:
            # Setup audio stream
            self._setup_audio_stream(device_id)
            
            while True:
                # Wait for keypress
                event = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:
                    
                    if event.name == 'space':
                        if not self.is_recording:
                            self._start_recording()
                            print(f"\n🔴 Recording sample #{sample_count}... (Press SPACE to stop)")
                        else:
                            duration = self._stop_recording()
                            filename = output_path / f"{technique}_sample_{sample_count:03d}.wav"
                            self._save_recording(filename)
                            print(f"✅ Saved: {filename.name} ({duration:.1f}s)")
                            sample_count += 1
                            print("\nPress SPACE for next recording, ESC to exit...")
                    
                    elif event.name == 'esc':
                        if self.is_recording:
                            self._stop_recording()
                        break
                    
                    elif event.name == 'd':
                        if not self.is_recording:
                            # Delete last recorded file
                            last_file = output_path / f"{technique}_sample_{sample_count-1:03d}.wav"
                            if last_file.exists():
                                last_file.unlink()
                                sample_count -= 1
                                print(f"❌ Deleted: {last_file.name}")
                            else:
                                print("No file to delete")
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self._cleanup()
            print(f"\nRecording session finished. Recorded {sample_count-1} samples.")
    
    def _setup_audio_stream(self, device_id):
        """Setup PyAudio stream"""
        try:
            self.stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            
        except Exception as e:
            self.logger.error(f"Failed to setup audio stream: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for recording"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data.copy())
        
        return (None, pyaudio.paContinue)
    
    def _start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.recorded_data = []
        self.start_time = time.time()
        
        # Clear any old data in queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
    
    def _stop_recording(self) -> float:
        """Stop recording and return duration"""
        self.is_recording = False
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Collect all recorded data
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            self.recorded_data.append(chunk)
        
        return duration
    
    def _save_recording(self, filename):
        """Save recorded audio to file"""
        if not self.recorded_data:
            print("No audio data to save")
            return
        
        # Combine all chunks
        audio_data = np.concatenate(self.recorded_data)
        
        # Save to WAV file
        sf.write(filename, audio_data, self.sample_rate)
    
    def _cleanup(self):
        """Cleanup audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.pa.terminate()
    
    def record_batch_samples(self, output_dir: str, technique: str, count: int, 
                           duration: float, device_id: int = None):
        """Record multiple samples with prompts"""
        output_path = Path(output_dir) / technique
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== Recording {count} samples of {technique.upper()} ===")
        print(f"Duration: {duration} seconds each")
        print(f"Output: {output_path}")
        
        try:
            self._setup_audio_stream(device_id)
            
            for i in range(count):
                print(f"\nSample {i+1}/{count}")
                print("Get ready... Recording starts in 3 seconds")
                time.sleep(3)
                
                print("🔴 Recording...")
                self._start_recording()
                time.sleep(duration)
                self._stop_recording()
                
                filename = output_path / f"{technique}_sample_{i+1:03d}.wav"
                self._save_recording(filename)
                print(f"✅ Saved: {filename.name}")
                
                if i < count - 1:
                    print("Take a break, next sample in 5 seconds...")
                    time.sleep(5)
        
        finally:
            self._cleanup()


def main():
    parser = argparse.ArgumentParser(description="Collect training samples for technique detection")
    
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio input devices')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive recording session with keyboard controls')
    
    parser.add_argument('--batch', action='store_true',
                       help='Batch recording mode with automatic timing')
    
    parser.add_argument('--output-dir', type=str, default='training_samples',
                       help='Output directory for samples (default: training_samples)')
    
    parser.add_argument('--technique', type=str, 
                       choices=['none', 'chugging', 'harmonic'],
                       help='Technique to record')
    
    parser.add_argument('--count', type=int, default=10,
                       help='Number of samples to record in batch mode')
    
    parser.add_argument('--duration', type=float, default=3.0,
                       help='Duration of each sample in seconds (batch mode)')
    
    parser.add_argument('--device', type=int,
                       help='Audio input device ID (use --list-devices to see options)')
    
    args = parser.parse_args()
    
    recorder = SampleRecorder()
    
    if args.list_devices:
        recorder.list_audio_devices()
        return
    
    if not args.technique:
        print("Please specify --technique (none, chugging, or harmonic)")
        return 1
    
    if args.interactive:
        print("=== Interactive Recording Mode ===")
        print("This mode lets you control recording with keyboard shortcuts")
        recorder.start_recording_session(args.output_dir, args.technique, args.device)
    
    elif args.batch:
        print("=== Batch Recording Mode ===")
        print("This mode records a fixed number of samples with automatic timing")
        recorder.record_batch_samples(
            args.output_dir, args.technique, args.count, args.duration, args.device
        )
    
    else:
        print("Please specify either --interactive or --batch mode")
        print("Use --interactive for flexible recording with keyboard controls")
        print("Use --batch for automated timing")
        return 1


if __name__ == "__main__":
    exit(main())