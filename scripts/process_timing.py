#!/usr/bin/env python3
"""
Process timing CSV files to convert GPU cycles to nanoseconds and analyze results.

Usage:
    python scripts/process_timing.py <timing_csv_file> [--gpu-freq-ghz <freq>]
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def get_gpu_clock_frequency():
    """
    Try to get GPU clock frequency from nvidia-smi.
    Returns frequency in GHz, or None if not available.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        # Returns clock in MHz
        freq_mhz = float(result.stdout.strip())
        return freq_mhz / 1000.0  # Convert to GHz
    except Exception as e:
        print(f"Warning: Could not auto-detect GPU frequency: {e}", file=sys.stderr)
        return None

def process_timing_file(input_file, gpu_freq_ghz=None):
    """
    Process a timing CSV file and convert GPU cycles to nanoseconds.
    
    Args:
        input_file: Path to input CSV file
        gpu_freq_ghz: GPU clock frequency in GHz (if None, will try to auto-detect)
    
    Returns:
        Processed DataFrame
    """
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Auto-detect GPU frequency if not provided
    if gpu_freq_ghz is None:
        gpu_freq_ghz = get_gpu_clock_frequency()
        if gpu_freq_ghz is None:
            print("Error: GPU frequency not provided and could not be auto-detected.", file=sys.stderr)
            print("Please provide frequency with --gpu-freq-ghz", file=sys.stderr)
            sys.exit(1)
    
    print(f"Using GPU frequency: {gpu_freq_ghz} GHz")
    
    # Cycle period in nanoseconds
    ns_per_cycle = 1.0 / gpu_freq_ghz
    
    # Add converted columns for GPU entries (explicitly as float64)
    df['duration_ns'] = df['duration'].astype('float64')
    df['wait_time_ns'] = df['wait_time'].astype('float64')
    df['duration_corrected_ns'] = df['duration_corrected'].astype('float64')
    df['buffer_read_time_ns'] = df['buffer_read_time'].astype('float64')
    df['per_element_read_time_ns'] = df['per_element_read_time'].astype('float64')
    
    # Convert GPU cycles to nanoseconds
    gpu_mask = df['device'] == 'gpu'
    df.loc[gpu_mask, 'duration_ns'] = df.loc[gpu_mask, 'duration'].astype('float64') * ns_per_cycle
    df.loc[gpu_mask, 'wait_time_ns'] = df.loc[gpu_mask, 'wait_time'].astype('float64') * ns_per_cycle
    df.loc[gpu_mask, 'duration_corrected_ns'] = df.loc[gpu_mask, 'duration_corrected'].astype('float64') * ns_per_cycle
    df.loc[gpu_mask, 'buffer_read_time_ns'] = df.loc[gpu_mask, 'buffer_read_time'].astype('float64') * ns_per_cycle
    df.loc[gpu_mask, 'per_element_read_time_ns'] = df.loc[gpu_mask, 'per_element_read_time'].astype('float64') * ns_per_cycle
    
    return df

def print_summary(df):
    """Print summary statistics of timing data."""
    print("\n" + "="*80)
    print("TIMING SUMMARY (all values in nanoseconds)")
    print("="*80)
    
    for device in df['device'].unique():
        device_df = df[df['device'] == device]
        print(f"\n{device.upper()} Threads:")
        print("-" * 80)
        
        for role in device_df['role'].unique():
            role_df = device_df[device_df['role'] == role]
            
            if len(role_df) == 0:
                continue
            
            # Group by caching status if applicable
            if 'caching' in role_df.columns and role_df['caching'].nunique() > 1:
                for caching in sorted(role_df['caching'].unique()):
                    cache_df = role_df[role_df['caching'] == caching]
                    cache_label = "CACHED" if caching in ['true', 'True', True] else "NO-CACHE"
                    
                    print(f"\n  {role} ({cache_label}):")
                    print(f"    Count: {len(cache_df)}")
                    print_role_stats(cache_df)
            else:
                print(f"\n  {role}:")
                print(f"    Count: {len(role_df)}")
                print_role_stats(role_df)
    
    print("\n" + "="*80)

def print_role_stats(role_df):
    """Print statistics for a role dataframe."""
    print(f"    Wait time (ns):")
    print(f"      Mean:   {role_df['wait_time_ns'].mean():.2f}")
    print(f"      Median: {role_df['wait_time_ns'].median():.2f}")
    print(f"      Min:    {role_df['wait_time_ns'].min():.2f}")
    print(f"      Max:    {role_df['wait_time_ns'].max():.2f}")
    print(f"      Std:    {role_df['wait_time_ns'].std():.2f}")
    
    print(f"    Duration corrected (ns):")
    print(f"      Mean:   {role_df['duration_corrected_ns'].mean():.2f}")
    print(f"      Median: {role_df['duration_corrected_ns'].median():.2f}")
    print(f"      Min:    {role_df['duration_corrected_ns'].min():.2f}")
    print(f"      Max:    {role_df['duration_corrected_ns'].max():.2f}")
    print(f"      Std:    {role_df['duration_corrected_ns'].std():.2f}")
    
    print(f"    Buffer read time (ns):")
    print(f"      Mean:   {role_df['buffer_read_time_ns'].mean():.2f}")
    print(f"      Median: {role_df['buffer_read_time_ns'].median():.2f}")
    print(f"      Min:    {role_df['buffer_read_time_ns'].min():.2f}")
    print(f"      Max:    {role_df['buffer_read_time_ns'].max():.2f}")
    print(f"      Std:    {role_df['buffer_read_time_ns'].std():.2f}")
    
    print(f"    Per-element read time (ns):")
    print(f"      Mean:   {role_df['per_element_read_time_ns'].mean():.4f}")
    print(f"      Median: {role_df['per_element_read_time_ns'].median():.4f}")
    print(f"      Min:    {role_df['per_element_read_time_ns'].min():.4f}")
    print(f"      Max:    {role_df['per_element_read_time_ns'].max():.4f}")
    print(f"      Std:    {role_df['per_element_read_time_ns'].std():.4f}")
    
    # Check final values (if column exists)
    if 'final_value' in role_df.columns:
        unique_values = role_df['final_value'].unique()
        if len(unique_values) <= 10:
            print(f"    Final values: {sorted(unique_values)}")
        else:
            print(f"    Final values: {len(unique_values)} unique values")

def compare_orderings(df):
    """Compare acquire vs relaxed orderings."""
    print("\n" + "="*80)
    print("ACQUIRE vs RELAXED COMPARISON")
    print("="*80)
    
    for device in df['device'].unique():
        device_df = df[df['device'] == device]
        
        # Check if caching column exists
        has_caching = 'caching' in device_df.columns
        
        if has_caching and device_df['caching'].nunique() > 1:
            # Compare by caching status
            for caching in sorted(device_df['caching'].unique()):
                cache_df = device_df[device_df['caching'] == caching]
                cache_label = "CACHED" if caching in ['true', 'True', True] else "NO-CACHE"
                
                acquire_df = cache_df[cache_df['ordering'] == 'acquire']
                relaxed_df = cache_df[cache_df['ordering'] == 'relaxed']
                
                if len(acquire_df) > 0 and len(relaxed_df) > 0:
                    print(f"\n{device.upper()} ({cache_label}):")
                    print_ordering_comparison(acquire_df, relaxed_df)
        else:
            # Compare without caching breakdown
            acquire_df = device_df[device_df['ordering'] == 'acquire']
            relaxed_df = device_df[device_df['ordering'] == 'relaxed']
            
            if len(acquire_df) > 0 and len(relaxed_df) > 0:
                print(f"\n{device.upper()}:")
                print_ordering_comparison(acquire_df, relaxed_df)
    
    # Compare by scope if watch_flag column exists
    if 'watch_flag' in df.columns:
        print("\n" + "="*80)
        print("COMPARISON BY SCOPE")
        print("="*80)
        
        for scope in sorted(df['watch_flag'].unique()):
            if scope == 'N/A':
                continue
            scope_df = df[df['watch_flag'] == scope]
            
            if has_caching and scope_df['caching'].nunique() > 1:
                for caching in sorted(scope_df['caching'].unique()):
                    cache_df = scope_df[scope_df['caching'] == caching]
                    cache_label = "CACHED" if caching in ['true', 'True', True] else "NO-CACHE"
                    
                    if len(cache_df) > 0:
                        print(f"\n{scope.upper()} scope ({cache_label}):")
                        print(f"  Count: {len(cache_df)}")
                        print(f"  Buffer read time (ns): {cache_df['buffer_read_time_ns'].mean():.2f} ± {cache_df['buffer_read_time_ns'].std():.2f}")
                        print(f"  Per-element time (ns): {cache_df['per_element_read_time_ns'].mean():.4f} ± {cache_df['per_element_read_time_ns'].std():.4f}")
            else:
                if len(scope_df) > 0:
                    print(f"\n{scope.upper()} scope:")
                    print(f"  Count: {len(scope_df)}")
                    print(f"  Buffer read time (ns): {scope_df['buffer_read_time_ns'].mean():.2f} ± {scope_df['buffer_read_time_ns'].std():.2f}")
                    print(f"  Per-element time (ns): {scope_df['per_element_read_time_ns'].mean():.4f} ± {scope_df['per_element_read_time_ns'].std():.4f}")

def print_ordering_comparison(acquire_df, relaxed_df):
    """Print comparison between acquire and relaxed orderings."""
    print(f"  Acquire wait time (ns): {acquire_df['wait_time_ns'].mean():.2f} ± {acquire_df['wait_time_ns'].std():.2f}")
    print(f"  Relaxed wait time (ns): {relaxed_df['wait_time_ns'].mean():.2f} ± {relaxed_df['wait_time_ns'].std():.2f}")
    diff = acquire_df['wait_time_ns'].mean() - relaxed_df['wait_time_ns'].mean()
    rel_mean = relaxed_df['wait_time_ns'].mean()
    if rel_mean > 0:
        print(f"  Difference: {diff:.2f} ns ({(diff/rel_mean*100):.2f}%)")
    
    print(f"  Acquire buffer read (ns): {acquire_df['buffer_read_time_ns'].mean():.2f} ± {acquire_df['buffer_read_time_ns'].std():.2f}")
    print(f"  Relaxed buffer read (ns): {relaxed_df['buffer_read_time_ns'].mean():.2f} ± {relaxed_df['buffer_read_time_ns'].std():.2f}")
    diff_read = acquire_df['buffer_read_time_ns'].mean() - relaxed_df['buffer_read_time_ns'].mean()
    rel_read_mean = relaxed_df['buffer_read_time_ns'].mean()
    if rel_read_mean > 0:
        print(f"  Difference: {diff_read:.2f} ns ({(diff_read/rel_read_mean*100):.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Process timing CSV files from cache invalidation testing')
    parser.add_argument('input_file', type=str, help='Input timing CSV file')
    parser.add_argument('--gpu-freq-ghz', type=float, default=None,
                        help='GPU clock frequency in GHz (default: auto-detect)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file (default: <input>_processed.csv)')
    parser.add_argument('--no-summary', action='store_true',
                        help='Skip printing summary statistics')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Process file
    print(f"Processing: {args.input_file}")
    df = process_timing_file(args.input_file, args.gpu_freq_ghz)
    
    # Determine output filename
    if args.output is None:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_processed.csv"
    else:
        output_file = args.output
    
    # Save processed file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")
    
    # Print summary
    if not args.no_summary:
        print_summary(df)
        compare_orderings(df)
    
    print(f"\nDone!")

if __name__ == '__main__':
    main()
