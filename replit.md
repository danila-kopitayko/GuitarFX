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
The application uses an optimized multi-threaded architecture to ensure real-time performance:

- Main thread handles GUI operations and user interaction (optimized to 2Hz updates)
- Dedicated audio processing thread manages the core audio pipeline (10ms sleep cycles)
- Separate visualization thread updates displays at 5Hz (reduced from 30Hz to prevent blocking)
- Thread-safe buffer management prevents audio dropouts
- Lightweight GUI updates prevent thread contention and blocking

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

## Enhanced AI Training System (Added August 2025)

### Improved Technique Detection Training
The system now includes comprehensive training tools to improve AI accuracy in distinguishing between guitar techniques, especially chugging vs pinch harmonics:

- **Enhanced Synthetic Data**: More realistic audio generation that captures actual acoustic properties of techniques
- **Real Recording Training**: Ability to train from real guitar recordings organized in folders
- **Training Sample Collection**: Interactive recording tools to collect technique samples
- **Model Evaluation**: Detailed performance analysis and feature importance

### Training Commands

#### Train with Improved Synthetic Data Only
```bash
python train_technique_model.py --synthetic-only
```

#### Train with Real WAV Recordings
```bash
# First, create the directory structure
python train_technique_model.py --create-structure training_data

# Add your WAV files to: training_data/chugging/, training_data/harmonic/, training_data/none/
# Then train the model
python train_technique_model.py --wav-data training_data
```

#### Train with Both Synthetic and Real Data
```bash
python train_technique_model.py --wav-data training_data --combined
```

### Recording Training Samples

#### Interactive Recording (Recommended)
```bash
# List available audio input devices
python collect_training_samples.py --list-devices

# Start interactive recording session with keyboard controls
python collect_training_samples.py --interactive --technique chugging --device 2
python collect_training_samples.py --interactive --technique harmonic --device 2
python collect_training_samples.py --interactive --technique none --device 2
```

#### Batch Recording
```bash
# Record 20 samples of 4 seconds each
python collect_training_samples.py --batch --technique chugging --count 20 --duration 4.0
```

### Interactive Recording Controls
- **SPACE**: Start/Stop recording
- **ESC**: Exit recording session  
- **D**: Delete last recording

### Training Improvements
The enhanced training system addresses the key acoustic differences between techniques:

**Chugging Characteristics:**
- Palm-muted attack patterns with sharp onset and quick decay
- Limited harmonic content due to palm muting
- Low frequency emphasis (80-200 Hz range)
- Fret buzz and string noise from aggressive playing
- Rhythmic patterns and percussive nature

**Pinch Harmonic Characteristics:**  
- High harmonic content with specific frequency nodes emphasized
- Sharp attack followed by sustained decay
- Squealing frequencies from harmonic series (2nd, 3rd, 5th, 7th, 12th frets)
- Subtle vibrato and pitch bending
- Extended sustain compared to chugging

### Model Performance
The enhanced model provides:
- **Better Feature Extraction**: 23 different audio features optimized for guitar techniques
- **Improved Classification**: RandomForest with 200 trees and balanced classes
- **Real-time Evaluation**: Confusion matrix and feature importance analysis
- **Validation**: Train/test split with detailed performance metrics

## Performance Optimizations (August 2025)

### GUI Freezing Issue Resolution
Resolved critical application freezing issue through comprehensive performance optimizations:

#### Problem Identified
- **GUI Thread Blocking**: Heavy matplotlib operations in visualizer were blocking the main GUI thread
- **Excessive Update Frequency**: GUI updates at 10Hz and visualizer at 30Hz caused CPU overload
- **Expensive Computations**: Real-time spectrum analysis and spectrogram calculations were too intensive
- **Threading Contention**: Multiple threads competing for GUI resources

#### Optimizations Applied

**Visualizer Performance Fixes:**
- Reduced update frequency from 30Hz to 5Hz to prevent CPU overload
- Shortened data history from 200 to 50 frames to reduce memory usage
- Removed expensive FFT spectrum analysis and spectrogram calculations
- Implemented lightweight audio processing that skips heavy computations
- Added non-blocking matplotlib canvas updates with error handling

**GUI Thread Optimizations:**
- Reduced main GUI update frequency from 10Hz to 2Hz
- Implemented minimal status updates that only change when necessary
- Removed expensive control panel updates from main thread loop
- Added lightweight update methods that prevent GUI blocking

**Audio Processing Improvements:**
- Increased processing loop sleep time from 5ms to 10ms for better thread balance
- Extended error recovery sleep times to prevent rapid error cycles
- Optimized buffer management to reduce thread contention

#### Results
- **Eliminated Freezing**: Application now runs smoothly without GUI blocking
- **Improved Responsiveness**: Interface remains responsive during audio processing
- **Reduced CPU Usage**: Significantly lower CPU overhead from GUI operations
- **Stable Performance**: No more buffer underrun warnings or frequent error cycles

#### Technical Changes
- `gui/visualizer.py`: Simplified update loop, removed heavy matplotlib operations
- `gui/main_window.py`: Reduced update frequency, minimal status updates only
- `gui/control_panel.py`: Lightweight updates that check for changes before updating
- `main.py`: Optimized audio processing loop timing for better thread balance

These optimizations ensure the application runs smoothly in real-time without the freezing issues that were preventing normal usage.