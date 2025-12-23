#!/usr/bin/env python3
"""
Δt Hallucination Detector - Main Entry Point
Easy invocation for detection, testing, and experiments
"""

import argparse
import sys
import os

# Ensure the detector package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_detection(args):
    """Run detection on a single prompt or file of prompts"""
    from detector import DeltaTDetector, format_console_report
    
    print(f"Initializing detector with model: {args.model}")
    print(f"Device preference: {args.device}")
    
    # Update config for device
    from detector.config import DetectorConfig
    config = DetectorConfig()
    config.device = args.device
    
    detector = DeltaTDetector(model_name=args.model, config=config)
    detector.set_risk_profile(args.profile)
    
    print(f"Risk profile: {args.profile}")
    print("-" * 50)
    
    if args.prompt:
        # Single prompt
        prompts = [args.prompt]
    elif args.file:
        # File of prompts (one per line)
        with open(args.file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide either --prompt or --file")
        sys.exit(1)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        if args.multi_invariant:
            result = detector.detect_multi_invariant(
                prompt,
                validate_citations=args.validate_citations,
                test_semantic=args.test_semantic
            )
        else:
            result = detector.detect(prompt)
        
        if args.verbose and result.report:
            print(format_console_report(result.report))
        else:
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Temporal debt: {result.temporal_debt:.3f}")


def run_tests(args):
    """Run the test suite"""
    print("Running Δt Detector test suite...")
    print("-" * 50)
    
    # Try pytest first, fall back to unittest
    try:
        import pytest
        test_path = os.path.join(os.path.dirname(__file__), 'tests.py')
        
        pytest_args = [test_path, '-v']
        if args.quick:
            pytest_args.extend(['-x'])  # Stop on first failure
        if args.coverage:
            pytest_args.extend(['--cov=detector', '--cov-report=term-missing'])
        
        sys.exit(pytest.main(pytest_args))
        
    except ImportError:
        print("pytest not installed, falling back to unittest...")
        import unittest
        
        # Import test module
        from detector import tests
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(tests)
        
        verbosity = 2 if args.verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        sys.exit(0 if result.wasSuccessful() else 1)


def run_experiment(args):
    """Run TruthfulQA validation experiment"""
    from detector.truthfulqa_experiment import run_experiment, compare_risk_profiles
    
    if args.compare_profiles:
        compare_risk_profiles(num_samples=args.samples)
    else:
        run_experiment(
            num_samples=args.samples,
            risk_profile=args.profile,
            create_baseline=not args.no_baseline,
            use_multi_invariant=args.multi_invariant,
            model_name=args.model
        )


def show_device_info(args):
    """Show available compute devices"""
    print("Δt Detector - Device Information")
    print("=" * 50)
    
    # Check torch availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print()
        
        # CUDA
        print("CUDA (NVIDIA GPU):")
        if torch.cuda.is_available():
            print(f"  ✓ Available")
            print(f"  Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    [{i}] {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            print("  ✗ Not available")
        print()
        
        # MPS (Apple Silicon)
        print("MPS (Apple Silicon):")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✓ Available")
            if torch.backends.mps.is_built():
                print(f"  PyTorch built with MPS support")
        else:
            print("  ✗ Not available")
        print()
        
        # CPU
        print("CPU:")
        print(f"  ✓ Always available (fallback)")
        print()
        
        # Recommendation
        print("Recommended device for this system:")
        if torch.cuda.is_available():
            print("  → cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  → mps")
        else:
            print("  → cpu")
            
    except ImportError:
        print("PyTorch not installed!")
        print("Install with: pip install torch")


def main():
    parser = argparse.ArgumentParser(
        description="Δt Hallucination Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check available devices
  python -m detector.run devices
  
  # Run tests
  python -m detector.run test
  
  # Detect single prompt
  python -m detector.run detect --prompt "What is the capital of France?"
  
  # Detect with medical profile
  python -m detector.run detect --prompt "..." --profile medical
  
  # Run TruthfulQA experiment
  python -m detector.run experiment --samples 200
  
  # Use specific device
  python -m detector.run detect --prompt "..." --device mps
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run detection on prompts')
    detect_parser.add_argument('--prompt', type=str, help='Single prompt to analyze')
    detect_parser.add_argument('--file', type=str, help='File with prompts (one per line)')
    detect_parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                               help='Model to use')
    detect_parser.add_argument('--profile', type=str, default='general',
                               choices=['medical', 'legal', 'research', 'general', 'creative', 'entertainment'],
                               help='Risk profile')
    detect_parser.add_argument('--device', type=str, default='auto',
                               choices=['auto', 'cuda', 'mps', 'cpu'],
                               help='Compute device')
    detect_parser.add_argument('--multi-invariant', action='store_true',
                               help='Use full multi-invariant detection')
    detect_parser.add_argument('--validate-citations', action='store_true',
                               help='Validate URLs/DOIs (requires network)')
    detect_parser.add_argument('--test-semantic', action='store_true',
                               help='Test semantic conservation')
    detect_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show full report')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    test_parser.add_argument('--quick', '-q', action='store_true',
                             help='Stop on first failure')
    test_parser.add_argument('--coverage', action='store_true',
                             help='Run with coverage (requires pytest-cov)')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run TruthfulQA experiment')
    exp_parser.add_argument('--samples', type=int, default=400,
                            help='Number of samples')
    exp_parser.add_argument('--profile', type=str, default='general',
                            help='Risk profile')
    exp_parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                            help='Model to use')
    exp_parser.add_argument('--multi-invariant', action='store_true',
                            help='Use multi-invariant detection')
    exp_parser.add_argument('--no-baseline', action='store_true',
                            help='Skip baseline calibration')
    exp_parser.add_argument('--compare-profiles', action='store_true',
                            help='Compare multiple risk profiles')
    
    # Devices command
    devices_parser = subparsers.add_parser('devices', help='Show available compute devices')
    
    args = parser.parse_args()
    
    if args.command == 'detect':
        run_detection(args)
    elif args.command == 'test':
        run_tests(args)
    elif args.command == 'experiment':
        run_experiment(args)
    elif args.command == 'devices':
        show_device_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
