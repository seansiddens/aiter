import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

def parse_mha_file(filename):
    """Parse the MHA benchmark file and return a DataFrame."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    # Find where the data starts (after the header line with column names)
    data_start = 0
    for i, line in enumerate(lines):
        if 'BATCH' in line and 'HQ' in line:
            data_start = i + 1
            break
    
    # Parse the data lines
    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 6:
                data.append({
                    'index': int(parts[0]),
                    'BATCH': float(parts[1]),
                    'HQ': float(parts[2]),
                    'HK': float(parts[3]),
                    'N_CTX_Q': float(parts[4]),
                    'N_CTX_K': float(parts[5]),
                    'fwd_TFLOPS': float(parts[6])
                })
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Compare MHA TFLOPS performance between two benchmark files.')
    parser.add_argument('file1', help='Path to first MHA benchmark file')
    parser.add_argument('file2', help='Path to second MHA benchmark file')
    parser.add_argument('-o', '--output', default='mha_tflops_comparison.png',
                        help='Output filename for the plot (default: mha_tflops_comparison.png)')
    
    args = parser.parse_args()
    
    # Extract basenames for legend labels (remove extension)
    label1 = os.path.splitext(os.path.basename(args.file1))[0]
    label2 = os.path.splitext(os.path.basename(args.file2))[0]
    
    # Load both files
    print(f"Loading file 1: {args.file1}")
    df1 = parse_mha_file(args.file1)
    
    print(f"Loading file 2: {args.file2}")
    df2 = parse_mha_file(args.file2)
    
    # Merge on index to ensure we're comparing the same configurations
    merged_df = df1.merge(df2, on='index', suffixes=('_file1', '_file2'))
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(merged_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, merged_df['fwd_TFLOPS_file1'], width, 
                   label=label1, alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, merged_df['fwd_TFLOPS_file2'], width,
                   label=label2, alpha=0.8, color='#ff7f0e')
    
    # Customize the plot
    ax.set_xlabel('Configuration Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('TFLOPS', fontsize=12, fontweight='bold')
    ax.set_title(f'MHA Performance Comparison: {label1} vs {label2}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(merged_df['index'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars for easier reading (only for selected bars to avoid clutter)
    for i in range(0, len(x), 3):  # Show every 3rd bar to avoid overcrowding
        height1 = merged_df['fwd_TFLOPS_file1'].iloc[i]
        height2 = merged_df['fwd_TFLOPS_file2'].iloc[i]
        if height1 > 100 or height2 > 100:  # Only label significant values
            ax.text(x[i] - width/2, height1, f'{height1:.0f}', 
                    ha='center', va='bottom', fontsize=7, rotation=90)
            ax.text(x[i] + width/2, height2, f'{height2:.0f}',
                    ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')

    print(f"\nPlot saved as '{args.output}'")
    
    # Also create a summary statistics table
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"{label1} - Mean: {merged_df['fwd_TFLOPS_file1'].mean():.2f} TFLOPS")
    print(f"{label2} - Mean: {merged_df['fwd_TFLOPS_file2'].mean():.2f} TFLOPS")
    print(f"\n{label1} - Max: {merged_df['fwd_TFLOPS_file1'].max():.2f} TFLOPS")
    print(f"{label2} - Max: {merged_df['fwd_TFLOPS_file2'].max():.2f} TFLOPS")
    
    # Calculate speedup/slowdown
    merged_df['speedup'] = merged_df['fwd_TFLOPS_file2'] / merged_df['fwd_TFLOPS_file1']
    print(f"\nAverage Speedup ({label2}/{label1}): {merged_df['speedup'].mean():.3f}x")
    print(f"Best case: {merged_df['speedup'].max():.3f}x (config {merged_df.loc[merged_df['speedup'].idxmax(), 'index']})")
    print(f"Worst case: {merged_df['speedup'].min():.3f}x (config {merged_df.loc[merged_df['speedup'].idxmin(), 'index']})")
    print("="*60)

if __name__ == '__main__':
    main()