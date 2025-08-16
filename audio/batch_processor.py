"""
Batch Audio Processor - Process WAV files through the guitar effects pipeline
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, List
import time

from utils.logger import Logger
from utils.buffer_manager import BufferManager
from detection.technique_detector import TechniqueDetector
from effects.effect_processor import EffectProcessor


class BatchAudioProcessor:
    """Process WAV files through the complete guitar effects pipeline"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger()
        
        # Initialize components
        self.technique_detector = TechniqueDetector(settings)
        self.effect_processor = EffectProcessor(settings)
        self.buffer_manager = BufferManager(settings)
        
        # Processing parameters
        self.sample_rate = settings.get('audio.sample_rate', 44100)
        self.buffer_size = settings.get('audio.buffer_size', 512)
        self.analysis_window_size = settings.get('buffer.analysis_window_size', 4096)
        
        self.logger.info("Batch audio processor initialized")
    
    def process_wav_file(self, input_path: str, output_path: str, 
                        enable_effects: bool = True, 
                        technique_override: Optional[str] = None) -> bool:
        """
        Process a single WAV file through the effects pipeline
        
        Args:
            input_path: Path to input WAV file
            output_path: Path for output WAV file
            enable_effects: Whether to apply effects (False = dry signal analysis only)
            technique_override: Force a specific technique instead of detecting
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Processing: {input_path} -> {output_path}")
            
            # Load audio file
            audio_data, file_sample_rate = librosa.load(input_path, sr=self.sample_rate, mono=True)
            
            if len(audio_data) == 0:
                self.logger.error("Empty audio file")
                return False
            
            self.logger.info(f"Loaded audio: {len(audio_data)} samples, {len(audio_data)/self.sample_rate:.2f} seconds")
            
            # Process audio in chunks
            processed_audio = self._process_audio_chunks(
                audio_data, enable_effects, technique_override
            )
            
            # Save processed audio
            sf.write(output_path, processed_audio, self.sample_rate)
            self.logger.info(f"Saved processed audio to: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing WAV file: {e}")
            return False
    
    def _process_audio_chunks(self, audio_data: np.ndarray, 
                            enable_effects: bool,
                            technique_override: Optional[str]) -> np.ndarray:
        """Process audio data in chunks through the pipeline"""
        
        processed_chunks = []
        detected_techniques = []
        confidences = []
        
        # Process in buffer-sized chunks
        for i in range(0, len(audio_data), self.buffer_size):
            chunk = audio_data[i:i + self.buffer_size]
            
            # Pad last chunk if needed
            if len(chunk) < self.buffer_size:
                chunk = np.pad(chunk, (0, self.buffer_size - len(chunk)))
            
            # Add to buffer manager
            self.buffer_manager.add_frame(chunk)
            
            # Process chunk
            processed_chunk, technique, confidence = self._process_single_chunk(
                chunk, enable_effects, technique_override
            )
            
            processed_chunks.append(processed_chunk)
            if technique:
                detected_techniques.append(technique)
                confidences.append(confidence)
        
        # Combine processed chunks
        processed_audio = np.concatenate(processed_chunks)
        
        # Log technique detection summary
        if detected_techniques:
            unique_techniques = list(set(detected_techniques))
            self.logger.info(f"Detected techniques: {unique_techniques}")
            
            for tech in unique_techniques:
                indices = [i for i, t in enumerate(detected_techniques) if t == tech]
                avg_confidence = np.mean([confidences[i] for i in indices])
                duration = len(indices) * self.buffer_size / self.sample_rate
                self.logger.info(f"  {tech}: {duration:.1f}s (avg confidence: {avg_confidence:.2%})")
        
        # Trim to original length
        original_length = len(audio_data)
        if len(processed_audio) > original_length:
            processed_audio = processed_audio[:original_length]
        
        return processed_audio
    
    def _process_single_chunk(self, chunk: np.ndarray, 
                            enable_effects: bool,
                            technique_override: Optional[str]) -> Tuple[np.ndarray, Optional[str], float]:
        """Process a single audio chunk"""
        
        technique = None
        confidence = 0.0
        
        # Get analysis window if available
        if self.buffer_manager.has_complete_window():
            analysis_window = self.buffer_manager.get_analysis_window()
            
            if analysis_window is not None:
                if technique_override:
                    technique = technique_override
                    confidence = 1.0
                else:
                    # Detect technique
                    technique, confidence = self.technique_detector.detect_technique(analysis_window)
                
                # Apply confidence threshold
                min_confidence = self.settings.get('detection.min_confidence', 0.7)
                if confidence < min_confidence:
                    technique = None
                    confidence = 0.0
        
        # Apply effects if enabled
        if enable_effects and technique:
            self.effect_processor.set_technique(technique, confidence)
            processed_chunk = self.effect_processor.process_audio(chunk)
        else:
            processed_chunk = chunk
        
        return processed_chunk, technique, confidence
    
    def analyze_wav_file(self, input_path: str) -> dict:
        """
        Analyze a WAV file for technique detection without processing
        
        Args:
            input_path: Path to input WAV file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            self.logger.info(f"Analyzing: {input_path}")
            
            # Load audio file
            audio_data, _ = librosa.load(input_path, sr=self.sample_rate, mono=True)
            
            if len(audio_data) == 0:
                return {'error': 'Empty audio file'}
            
            # Clear buffer manager
            self.buffer_manager.clear_buffers()
            
            # Analyze in chunks
            detections = []
            timestamps = []
            
            for i in range(0, len(audio_data), self.buffer_size):
                chunk = audio_data[i:i + self.buffer_size]
                timestamp = i / self.sample_rate
                
                if len(chunk) < self.buffer_size:
                    chunk = np.pad(chunk, (0, self.buffer_size - len(chunk)))
                
                self.buffer_manager.add_frame(chunk)
                
                if self.buffer_manager.has_complete_window():
                    analysis_window = self.buffer_manager.get_analysis_window()
                    
                    if analysis_window is not None:
                        technique, confidence = self.technique_detector.detect_technique(analysis_window)
                        
                        min_confidence = self.settings.get('detection.min_confidence', 0.7)
                        if confidence >= min_confidence:
                            detections.append({
                                'timestamp': timestamp,
                                'technique': technique,
                                'confidence': confidence
                            })
                            timestamps.append(timestamp)
            
            # Generate summary
            techniques_found = {}
            for detection in detections:
                tech = detection['technique']
                if tech not in techniques_found:
                    techniques_found[tech] = {
                        'count': 0,
                        'total_confidence': 0.0,
                        'duration': 0.0
                    }
                
                techniques_found[tech]['count'] += 1
                techniques_found[tech]['total_confidence'] += detection['confidence']
                techniques_found[tech]['duration'] += self.buffer_size / self.sample_rate
            
            # Calculate averages
            for tech in techniques_found:
                info = techniques_found[tech]
                info['avg_confidence'] = info['total_confidence'] / info['count']
                info['percentage'] = (info['duration'] / (len(audio_data) / self.sample_rate)) * 100
            
            analysis = {
                'file_info': {
                    'duration_seconds': len(audio_data) / self.sample_rate,
                    'sample_rate': self.sample_rate,
                    'samples': len(audio_data)
                },
                'detections': detections,
                'summary': techniques_found,
                'total_detections': len(detections)
            }
            
            self.logger.info(f"Analysis complete: {len(detections)} technique detections found")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing WAV file: {e}")
            return {'error': str(e)}
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         enable_effects: bool = True,
                         technique_override: Optional[str] = None) -> List[str]:
        """
        Process all WAV files in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path  
            enable_effects: Whether to apply effects
            technique_override: Force specific technique
            
        Returns:
            List of successfully processed files
        """
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            
            if not input_path.exists():
                self.logger.error(f"Input directory does not exist: {input_dir}")
                return []
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find WAV files
            wav_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))
            
            if not wav_files:
                self.logger.info(f"No WAV files found in: {input_dir}")
                return []
            
            self.logger.info(f"Found {len(wav_files)} WAV files to process")
            
            processed_files = []
            
            for input_file in wav_files:
                output_file = output_path / f"processed_{input_file.name}"
                
                if self.process_wav_file(str(input_file), str(output_file), 
                                       enable_effects, technique_override):
                    processed_files.append(str(input_file))
                else:
                    self.logger.warning(f"Failed to process: {input_file}")
            
            self.logger.info(f"Successfully processed {len(processed_files)} files")
            return processed_files
            
        except Exception as e:
            self.logger.error(f"Error processing directory: {e}")
            return []
    
    def create_technique_samples(self, output_dir: str, duration: float = 5.0) -> bool:
        """
        Generate sample WAV files for different techniques for testing
        
        Args:
            output_dir: Directory to save samples
            duration: Duration of each sample in seconds
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            sample_rate = self.sample_rate
            samples = int(duration * sample_rate)
            t = np.linspace(0, duration, samples)
            
            # Generate different technique samples
            techniques = {
                'clean': self._generate_clean_guitar(t, sample_rate),
                'chugging': self._generate_chugging(t, sample_rate),
                'harmonic': self._generate_pinch_harmonic(t, sample_rate)
            }
            
            for technique, audio in techniques.items():
                filename = output_path / f"sample_{technique}.wav"
                sf.write(filename, audio, sample_rate)
                self.logger.info(f"Created sample: {filename}")
            
            self.logger.info(f"Generated {len(techniques)} technique samples in: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating technique samples: {e}")
            return False
    
    def _generate_clean_guitar(self, t: np.ndarray, sample_rate: int) -> np.ndarray:
        """Generate clean guitar sound"""
        # Single notes with natural decay
        freq = 196.0  # G3
        signal = 0.4 * np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-t * 1.5)
        return signal * envelope + np.random.normal(0, 0.01, len(t))
    
    def _generate_chugging(self, t: np.ndarray, sample_rate: int) -> np.ndarray:
        """Generate chugging sound"""
        freq = 82.4  # Low E
        signal = np.sin(2 * np.pi * freq * t) * 0.5
        
        # Sharp attack/decay pattern
        chug_rate = 4  # chugs per second  
        envelope = np.zeros_like(t)
        chug_duration = 0.2
        
        for i in range(int(len(t) / sample_rate * chug_rate)):
            start_time = i / chug_rate
            start_idx = int(start_time * sample_rate)
            end_idx = int((start_time + chug_duration) * sample_rate)
            end_idx = min(end_idx, len(envelope))
            
            if start_idx < len(envelope):
                chug_env = np.exp(-np.linspace(0, 5, end_idx - start_idx))
                envelope[start_idx:end_idx] = chug_env
        
        return signal * envelope + np.random.normal(0, 0.02, len(t))
    
    def _generate_pinch_harmonic(self, t: np.ndarray, sample_rate: int) -> np.ndarray:
        """Generate pinch harmonic sound"""
        fundamental = 82.4
        # Emphasize high harmonics
        signal = 0.3 * np.sin(2 * np.pi * fundamental * 2 * t)  # Octave
        signal += 0.5 * np.sin(2 * np.pi * fundamental * 5 * t)  # 5th harmonic
        signal += 0.2 * np.sin(2 * np.pi * fundamental * 7 * t)  # 7th harmonic
        
        # Sharp decay for harmonic character
        envelope = np.exp(-t * 3.0)
        return signal * envelope + np.random.normal(0, 0.01, len(t))