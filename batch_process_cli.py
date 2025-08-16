#!/usr/bin/env python3
"""
Command Line Interface for Batch WAV Processing
Process WAV files through the guitar effects system for easy testing
"""

import argparse
import sys
from pathlib import Path

from audio.batch_processor import BatchAudioProcessor
from config.settings import Settings
from utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(
        description="Process WAV files through guitar effects with AI technique detection"
    )
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process single file
    process_parser = subparsers.add_parser('process', help='Process a single WAV file')
    process_parser.add_argument('input', help='Input WAV file path')
    process_parser.add_argument('output', help='Output WAV file path')
    process_parser.add_argument('--dry', action='store_true', help='Disable effects (dry signal only)')
    process_parser.add_argument('--technique', choices=['chugging', 'harmonic', 'none'], 
                               help='Force specific technique instead of detecting')
    
    # Process directory
    batch_parser = subparsers.add_parser('batch', help='Process all WAV files in directory')
    batch_parser.add_argument('input_dir', help='Input directory path')
    batch_parser.add_argument('output_dir', help='Output directory path')
    batch_parser.add_argument('--dry', action='store_true', help='Disable effects (dry signal only)')
    batch_parser.add_argument('--technique', choices=['chugging', 'harmonic', 'none'],
                             help='Force specific technique instead of detecting')
    
    # Analyze file
    analyze_parser = subparsers.add_parser('analyze', help='Analyze WAV file for technique detection')
    analyze_parser.add_argument('input', help='Input WAV file path')
    analyze_parser.add_argument('--detailed', action='store_true', help='Show detailed detection timeline')
    
    # Generate samples
    samples_parser = subparsers.add_parser('samples', help='Generate test WAV samples')
    samples_parser.add_argument('output_dir', help='Directory to save samples')
    samples_parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize processor
    try:
        settings = Settings()
        processor = BatchAudioProcessor(settings)
        logger = Logger()
        
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return 1
    
    # Execute command
    try:
        if args.command == 'process':
            return process_file(processor, args)
        elif args.command == 'batch':
            return process_batch(processor, args)
        elif args.command == 'analyze':
            return analyze_file(processor, args)
        elif args.command == 'samples':
            return generate_samples(processor, args)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


def process_file(processor: BatchAudioProcessor, args) -> int:
    """Process a single WAV file"""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not input_path.suffix.lower() == '.wav':
        print(f"Error: Input file must be a WAV file: {args.input}")
        return 1
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {args.input}")
    print(f"Output: {args.output}")
    if args.dry:
        print("Mode: Dry (no effects)")
    if args.technique:
        print(f"Technique: {args.technique} (forced)")
    print()
    
    success = processor.process_wav_file(
        str(input_path),
        str(output_path),
        enable_effects=not args.dry,
        technique_override=args.technique
    )
    
    if success:
        print(f"✓ Successfully processed: {args.output}")
        return 0
    else:
        print(f"✗ Failed to process: {args.input}")
        return 1


def process_batch(processor: BatchAudioProcessor, args) -> int:
    """Process directory of WAV files"""
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        return 1
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.dry:
        print("Mode: Dry (no effects)")
    if args.technique:
        print(f"Technique: {args.technique} (forced)")
    print()
    
    processed_files = processor.process_directory(
        str(input_dir),
        args.output_dir,
        enable_effects=not args.dry,
        technique_override=args.technique
    )
    
    print(f"\n✓ Successfully processed {len(processed_files)} files")
    
    if processed_files:
        print("Processed files:")
        for file_path in processed_files:
            print(f"  - {Path(file_path).name}")
    
    return 0


def analyze_file(processor: BatchAudioProcessor, args) -> int:
    """Analyze a WAV file for technique detection"""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not input_path.suffix.lower() == '.wav':
        print(f"Error: Input file must be a WAV file: {args.input}")
        return 1
    
    print(f"Analyzing: {args.input}")
    print()
    
    analysis = processor.analyze_wav_file(str(input_path))
    
    if 'error' in analysis:
        print(f"✗ Analysis failed: {analysis['error']}")
        return 1
    
    # Display results
    file_info = analysis['file_info']
    print(f"File Information:")
    print(f"  Duration: {file_info['duration_seconds']:.2f} seconds")
    print(f"  Sample Rate: {file_info['sample_rate']} Hz")
    print(f"  Samples: {file_info['samples']}")
    print()
    
    if analysis['total_detections'] == 0:
        print("No techniques detected above confidence threshold")
        return 0
    
    print(f"Technique Detection Summary:")
    print(f"  Total detections: {analysis['total_detections']}")
    print()
    
    summary = analysis['summary']
    for technique, info in summary.items():
        print(f"  {technique.upper()}:")
        print(f"    Duration: {info['duration']:.2f}s ({info['percentage']:.1f}%)")
        print(f"    Detections: {info['count']}")
        print(f"    Avg Confidence: {info['avg_confidence']:.1%}")
        print()
    
    # Detailed timeline if requested
    if args.detailed and analysis['detections']:
        print("Detection Timeline:")
        for detection in analysis['detections']:
            timestamp = detection['timestamp']
            technique = detection['technique']
            confidence = detection['confidence']
            print(f"  {timestamp:6.2f}s: {technique:10} ({confidence:.1%})")
    
    return 0


def generate_samples(processor: BatchAudioProcessor, args) -> int:
    """Generate test WAV samples"""
    output_dir = Path(args.output_dir)
    
    print(f"Generating test samples in: {args.output_dir}")
    print(f"Duration: {args.duration} seconds")
    print()
    
    success = processor.create_technique_samples(str(output_dir), args.duration)
    
    if success:
        print(f"✓ Successfully generated test samples in: {args.output_dir}")
        print("\nGenerated files:")
        for sample_file in output_dir.glob("sample_*.wav"):
            print(f"  - {sample_file.name}")
        return 0
    else:
        print(f"✗ Failed to generate samples")
        return 1


if __name__ == "__main__":
    exit(main())