# Guitar Effects Processor - AI Technique Detection

## Overview

A real-time guitar effects processor that uses AI-driven technique detection to automatically apply appropriate audio effects. The application detects specific guitar playing techniques (chugging, pinch harmonics) through machine learning analysis of audio features and applies corresponding cyberpunk-style distortion or vintage broken record effects. Built with Python, it provides a complete audio processing pipeline with real-time visualization and an intuitive GUI interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Audio Processing Pipeline
The system uses a modular audio processing architecture built around PyAudio for low-latency real-time audio I/O. The main processing pipeline consists of:

- **AudioManager**: Handles audio device management, stream configuration, and I/O operations using PyAudio with configurable sample rates (44.1kHz default) and buffer sizes (512 samples default)
- **AudioProcessor**: Coordinates the main processing pipeline, managing frame-by-frame audio processing and performance monitoring
- **BufferManager**: Provides thread-safe circular buffering with configurable overlap (50% default) for analysis windows (4096 samples)

### AI Technique Detection
The application implements machine learning-based technique detection using:

- **FeatureExtractor**: Extracts comprehensive audio features including MFCCs (13 coefficients), spectral features, RMS energy, and frequency band analysis for different playing techniques
- **TechniqueDetector**: Uses Random Forest classification with StandardScaler preprocessing to detect three technique classes: none, chugging, and harmonic (pinch harmonics)
- **Model Management**: Supports both pre-trained model loading and runtime model training with confidence thresholding (0.7 default)

### Effects Processing System
The effects system follows an object-oriented plugin architecture:

- **BaseEffect**: Abstract base class defining the common interface for all audio effects with standardized parameter management and state handling
- **DistortionEffect**: Cyberpunk-style digital distortion with drive, tone control, bit crushing, and analog saturation for aggressive chugging detection
- **BrokenRecordEffect**: Vintage vinyl-style glitch effect with scratch simulation, wow/flutter, crackle noise, and pitch bending for pinch harmonics
- **EffectProcessor**: Manages effect chains with automatic technique-to-effect mapping and crossfade transitions (50ms default)

### GUI Architecture
The interface uses Tkinter with a component-based design:

- **MainWindow**: Primary application window with threaded GUI updates and event handling
- **ControlPanel**: Audio device selection, processing controls, and real-time effect parameter adjustment
- **Visualizer**: Real-time audio visualization using Matplotlib with waveform, spectrum, and technique detection displays

### Configuration Management
Centralized configuration system with:

- **Settings**: JSON-based persistent configuration with platform-specific defaults for Windows/macOS/Linux
- **AudioConfig**: Platform-aware audio driver preferences (WASAPI/CoreAudio/ALSA) and device quality ratings
- **Logger**: Singleton logging system with file rotation and configurable verbosity levels

### Threading Model
The application uses a multi-threaded architecture to ensure real-time performance:

- Main thread handles GUI operations and user interaction
- Dedicated audio processing thread manages the core audio pipeline
- Separate visualization thread updates displays at 30Hz
- Thread-safe buffer management prevents audio dropouts

## External Dependencies

### Core Audio Processing
- **PyAudio**: Cross-platform audio I/O library for real-time audio stream management
- **NumPy**: Numerical computing for audio signal processing and feature extraction
- **SciPy**: Advanced signal processing functions for filtering and spectral analysis

### Machine Learning
- **scikit-learn**: Random Forest classifier and StandardScaler for technique detection
- **librosa**: Audio analysis library for MFCC extraction and spectral feature computation
- **joblib**: Model serialization and deserialization for persistent ML models

### GUI and Visualization
- **Tkinter**: Built-in Python GUI framework for the main interface
- **Matplotlib**: Real-time audio visualization with spectrum analysis and waveform displays

### Utility Libraries
- **pathlib**: Modern path handling for cross-platform file operations
- **threading**: Multi-threaded architecture for real-time audio processing
- **queue**: Thread-safe data structures for audio buffer management
- **collections.deque**: Efficient circular buffers for audio data storage

### System Integration
- **platform**: Platform detection for audio driver preferences and system-specific configurations
- **json**: Configuration file persistence and settings management
- **logging**: Comprehensive logging system with file rotation and multiple output targets

## WAV File Processing (Added August 2025)

### Batch Audio Processing
The system now supports processing WAV audio files for easier testing and development:

- **BatchAudioProcessor**: Core module for processing WAV files through the complete effects pipeline
- **Command Line Interface**: `batch_process_cli.py` provides easy-to-use commands for WAV processing
- **soundfile**: Added dependency for high-quality WAV file I/O operations

### Available Commands

#### Process Single WAV File
```bash
python batch_process_cli.py process input.wav output.wav
python batch_process_cli.py process input.wav output.wav --technique chugging  # Force technique
python batch_process_cli.py process input.wav output.wav --dry  # No effects, analysis only
```

#### Batch Process Directory
```bash
python batch_process_cli.py batch input_dir output_dir
python batch_process_cli.py batch input_dir output_dir --technique harmonic
```

#### Analyze WAV File
```bash
python batch_process_cli.py analyze input.wav
python batch_process_cli.py analyze input.wav --detailed  # Show timeline
```

#### Generate Test Samples
```bash
python batch_process_cli.py samples output_dir
python batch_process_cli.py samples output_dir --duration 10.0  # 10-second samples
```

### Benefits for Testing
- **Reproducible Testing**: Process the same audio file multiple times with different settings
- **Effect Comparison**: Compare dry vs processed audio to hear effect changes
- **Technique Validation**: Test AI detection accuracy on known guitar techniques
- **Development Workflow**: Iterate on effects without needing real-time audio hardware