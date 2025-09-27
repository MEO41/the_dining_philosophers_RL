#!/usr/bin/env python3
"""
Train all RL algorithms for dining philosophers problem
Usage: python train_all.py
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_training(algorithm: str):
    """Run training for a specific algorithm"""
    print(f"\n{'='*60}")
    print(f"Starting {algorithm.upper()} training at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run training subprocess
        result = subprocess.run(
            [sys.executable, "train/train_models.py", algorithm],
            capture_output=False,
            text=True,
            check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✓ {algorithm.upper()} training completed successfully")
        print(f"Duration: {duration/60:.1f} minutes")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {algorithm.upper()} training failed with exit code {e.returncode}")
        return False, 0
    except KeyboardInterrupt:
        print(f"\n⚠ {algorithm.upper()} training interrupted by user")
        return False, 0
    except Exception as e:
        print(f"\n✗ {algorithm.upper()} training failed: {e}")
        return False, 0


def main():
    """Main function to train all algorithms"""
    algorithms = ['ppo', 'a2c', 'sac', 'dqn']
    
    print("Starting training for all algorithms...")
    print(f"Algorithms to train: {', '.join(algorithms)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    total_start_time = time.time()
    
    for i, algorithm in enumerate(algorithms, 1):
        print(f"\n[{i}/{len(algorithms)}] Training {algorithm.upper()}...")
        
        success, duration = run_training(algorithm)
        results[algorithm] = {
            'success': success,
            'duration': duration
        }
        
        if not success:
            print(f"\nTraining failed for {algorithm}. Continue with next algorithm? (y/n): ", end="")
            response = input().lower().strip()
            if response in ['n', 'no']:
                print("Training stopped by user.")
                break
    
    # Summary
    total_duration = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nResults:")
    for algorithm, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        duration_str = f"({result['duration']/60:.1f}m)" if result['success'] else ""
        print(f"  {algorithm.upper()}: {status} {duration_str}")
    
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nCompleted: {successful}/{len(results)} algorithms")
    
    if successful == len(algorithms):
        print("🎉 All algorithms trained successfully!")
    elif successful > 0:
        print("⚠ Some algorithms completed successfully")
    else:
        print("❌ No algorithms completed successfully")


if __name__ == "__main__":
    main()