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
    
    for idx, prompt in enumerate(prompts):
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

        # Governor signal emission
        if args.signal_out:
            from detector.governor_signal import build_governor_signal, write_governor_signal
            signal = build_governor_signal(
                result,
                run_id=args.run_id,
                turn_id=args.turn_id,
                model_id=args.model,
                profile=args.profile,
            )
            if len(prompts) > 1:
                base, ext = os.path.splitext(args.signal_out)
                sig_path = f"{base}_{idx}{ext}"
            else:
                sig_path = args.signal_out
            write_governor_signal(signal, sig_path)
            print(f"  Signal written to {sig_path}")


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
    try:
        from detector.truthfulqa_experiment import run_experiment, compare_risk_profiles
    except ImportError:
        print("Error: TruthfulQA experiment module not found.")
        print("Missing file: detector/truthfulqa_experiment.py")
        sys.exit(1)
    
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


def run_eval(args):
    """Run evaluation harness on a JSONL corpus"""
    from detector.eval import run_eval
    run_dir = run_eval(
        jsonl_path=args.file,
        model=args.model,
        profile_name=args.profile,
        device=args.device,
        baseline=args.baseline,
        output_csv=args.output,
        store=not args.no_store,
        multi_invariant=args.multi_invariant,
    )
    if run_dir is not None:
        print(f"Run stored: {run_dir}", file=sys.stderr)


def run_eval_diff(args):
    """Diff two eval CSV runs"""
    from detector.eval_diff import diff_csv
    diff_csv(
        old_path=args.old,
        new_path=args.new,
        output_csv=args.output,
        confidence_delta=args.conf_delta
    )


def run_replay(args):
    """CPU-only threshold replay on a stored run"""
    import subprocess
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'scripts', 'replay.py')
    cmd = [sys.executable, script, '--run', args.run,
           '--accel-grid'] + [str(x) for x in args.accel_grid] + [
           '--tokens-grid'] + [str(x) for x in args.tokens_grid] + [
           '--slope-grid'] + [str(x) for x in args.slope_grid] + [
           '--sc-threshold', str(args.sc_threshold),
           '--min-violated'] + [str(x) for x in args.min_violated]
    if args.output:
        cmd += ['--output', args.output]
    sys.exit(subprocess.call(cmd))


def run_runs(args):
    """Manage stored eval runs"""
    from detector.run_store import list_runs, load_run, diff_runs, verify_run

    action = args.runs_action

    if action == "list":
        manifests = list_runs()
        if not manifests:
            print("No runs found.")
            return
        print(f"{'RUN ID':<14} {'FINISHED':<22} {'MODEL':<30} {'PROFILE':<12} {'SIZE':>5}")
        print("-" * 85)
        for m in manifests:
            print(
                f"{m['run_id']:<14} "
                f"{m['finished_at']:<22} "
                f"{m['model']:<30} "
                f"{m['profile']:<12} "
                f"{m['corpus_size']:>5}"
            )

    elif action == "show":
        run_dir = _resolve_run_dir(args.run_id)
        if run_dir is None:
            print(f"No run matching '{args.run_id}' found.", file=sys.stderr)
            sys.exit(1)
        data = load_run(run_dir)
        import json
        print("=== Manifest ===")
        print(json.dumps(data["manifest"], indent=2))
        print("\n=== Summary ===")
        print(json.dumps(data["summary"], indent=2))

    elif action == "diff":
        dir_a = _resolve_run_dir(args.run_a)
        dir_b = _resolve_run_dir(args.run_b)
        if dir_a is None:
            print(f"No run matching '{args.run_a}' found.", file=sys.stderr)
            sys.exit(1)
        if dir_b is None:
            print(f"No run matching '{args.run_b}' found.", file=sys.stderr)
            sys.exit(1)
        import json
        result = diff_runs(dir_a, dir_b)
        print(json.dumps(result, indent=2))

    elif action == "verify":
        if args.all:
            manifests = list_runs()
            if not manifests:
                print("No runs found.")
                return
            ok = 0
            fail = 0
            for m in manifests:
                rd = _resolve_run_dir(m["run_id"])
                if rd and verify_run(rd):
                    ok += 1
                else:
                    print(f"FAIL: {m['run_id']}")
                    fail += 1
            print(f"{ok} passed, {fail} failed")
        else:
            run_dir = _resolve_run_dir(args.run_id)
            if run_dir is None:
                print(f"No run matching '{args.run_id}' found.", file=sys.stderr)
                sys.exit(1)
            if verify_run(run_dir):
                print(f"OK: {args.run_id}")
            else:
                print(f"FAIL: {args.run_id}")
                sys.exit(1)


def _resolve_run_dir(prefix: str, runs_root: str = "runs") -> "str | None":
    """Find a run directory by run_id prefix match."""
    from pathlib import Path
    root = Path(runs_root)
    if not root.is_dir():
        return None
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(prefix):
            if (entry / "manifest.json").exists():
                return str(entry)
    return None


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
    detect_parser.add_argument('--signal-out', type=str, default=None,
                               help='Write governor signal JSON to this path')
    detect_parser.add_argument('--run-id', type=str, default='',
                               help='Run ID for governor signal')
    detect_parser.add_argument('--turn-id', type=str, default='',
                               help='Turn ID for governor signal')
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
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Run JSONL evaluation harness')
    eval_parser.add_argument('--file', type=str, required=True,
                             help='JSONL file with {id,prompt,expected_risk,notes}')
    eval_parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                             help='Model to use')
    eval_parser.add_argument('--profile', type=str, default='general',
                             choices=['medical', 'legal', 'research', 'general', 'creative', 'entertainment'],
                             help='Risk profile')
    eval_parser.add_argument('--device', type=str, default='auto',
                             choices=['auto', 'cuda', 'mps', 'cpu'],
                             help='Compute device')
    eval_parser.add_argument('--baseline', action='store_true',
                             help='Use baseline normalization if available')
    eval_parser.add_argument('--output', type=str, default=None,
                             help='Write CSV output to file (default: stdout)')
    eval_parser.add_argument('--no-store', action='store_true',
                             help='Skip storing run bundle under runs/')
    eval_parser.add_argument('--multi-invariant', action='store_true',
                             help='Use multi-invariant detection (Inv-1 + Inv-2)')

    # Eval diff command
    diff_parser = subparsers.add_parser('eval-diff', help='Diff two eval CSV runs')
    diff_parser.add_argument('--old', type=str, required=True, help='Old CSV file')
    diff_parser.add_argument('--new', type=str, required=True, help='New CSV file')
    diff_parser.add_argument('--output', type=str, default=None, help='Write CSV of changes')
    diff_parser.add_argument('--conf-delta', type=float, default=0.1,
                             help='Minimum confidence delta to count as change')
    
    # Replay command (CPU-only threshold scan)
    replay_parser = subparsers.add_parser('replay', help='CPU-only threshold replay on stored run')
    replay_parser.add_argument('--run', required=True, help='Run ID (prefix match)')
    replay_parser.add_argument('--accel-grid', type=float, nargs='+',
                               default=[0.25, 0.30, 0.35, 0.40, 0.45])
    replay_parser.add_argument('--tokens-grid', type=int, nargs='+',
                               default=[0, 2, 4, 6, 8])
    replay_parser.add_argument('--slope-grid', type=float, nargs='+',
                               default=[0.30, 0.35, 0.40])
    replay_parser.add_argument('--sc-threshold', type=float, default=0.70)
    replay_parser.add_argument('--min-violated', type=int, nargs='+', default=[1, 2])
    replay_parser.add_argument('--output', type=str, default=None,
                               help='Write results JSON to file')

    # Devices command
    devices_parser = subparsers.add_parser('devices', help='Show available compute devices')

    # Runs command
    runs_parser = subparsers.add_parser('runs', help='Manage stored eval runs')
    runs_sub = runs_parser.add_subparsers(dest='runs_action', help='Runs sub-action')
    runs_sub.required = True

    runs_list_parser = runs_sub.add_parser('list', help='List all stored runs')

    runs_show_parser = runs_sub.add_parser('show', help='Show manifest + summary for a run')
    runs_show_parser.add_argument('run_id', type=str, help='Run ID (prefix match)')

    runs_diff_parser = runs_sub.add_parser('diff', help='Diff two runs')
    runs_diff_parser.add_argument('run_a', type=str, help='First run ID (prefix match)')
    runs_diff_parser.add_argument('run_b', type=str, help='Second run ID (prefix match)')

    runs_verify_parser = runs_sub.add_parser('verify', help='Verify run integrity')
    runs_verify_parser.add_argument('run_id', type=str, nargs='?', default=None,
                                     help='Run ID (prefix match)')
    runs_verify_parser.add_argument('--all', action='store_true',
                                     help='Verify all runs')

    args = parser.parse_args()
    
    if args.command == 'detect':
        run_detection(args)
    elif args.command == 'test':
        run_tests(args)
    elif args.command == 'experiment':
        run_experiment(args)
    elif args.command == 'eval':
        run_eval(args)
    elif args.command == 'eval-diff':
        run_eval_diff(args)
    elif args.command == 'devices':
        show_device_info(args)
    elif args.command == 'replay':
        run_replay(args)
    elif args.command == 'runs':
        run_runs(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
