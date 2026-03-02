#!/usr/bin/env python3
"""
Create modified PURE STMaps for row analysis.

For each row index (0-24), creates a modified STMap dataset where only that specific row
is kept (all other rows set to zeros). This allows testing which row contributes most to
model performance.

Usage:
    python create_pure_trans_stmaps.py --row 0  # Generate row 0 only
    python create_pure_trans_stmaps.py --all     # Generate all 25 rows
"""
import os
import cv2
import numpy as np
import argparse
import shutil
from tqdm import tqdm

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEST_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
PURE_SOURCE_ROOT = os.path.join(NEST_ROOT, 'STMap', 'PURE')
PURE_TRANS_BASE = os.path.join(NEST_ROOT, 'STMap', 'PURE_trans_row')

# STMap filename options (try these in order)
STMAP_NAMES = ['STMap.png', 'STMap_RGB.png']


def find_stmap_file(subject_stmap_dir):
    """Find STMap file in subject directory. Returns (path, name) or (None, None)."""
    for name in STMAP_NAMES:
        path = os.path.join(subject_stmap_dir, name)
        if os.path.isfile(path):
            return path, name
    return None, None


def create_row_stmap(original_stmap, row_idx, keep_row=True):
    """
    Create modified STMap keeping only one row.
    
    Args:
        original_stmap: numpy array of shape (25, T, 3) - original STMap
        row_idx: row index to keep (0-24)
        keep_row: if True, keep row_idx and zero others; if False, zero row_idx and keep others
    
    Returns:
        Modified STMap array of same shape
    """
    stmap = original_stmap.copy()
    num_rows = stmap.shape[0]
    
    if row_idx < 0 or row_idx >= num_rows:
        raise ValueError(f"Row index {row_idx} out of range [0, {num_rows-1}]")
    
    if keep_row:
        # Keep only the specified row, zero all others
        mask = np.zeros(num_rows, dtype=bool)
        mask[row_idx] = True
        stmap[~mask] = 0
    else:
        # Zero the specified row, keep all others
        stmap[row_idx] = 0
    
    return stmap


def process_subject(subject_name, source_root, target_root, row_idx, stmap_filename):
    """
    Process one subject: copy Label folder and create modified STMap.
    
    Args:
        subject_name: subject folder name (e.g., '01-01')
        source_root: source PURE root directory
        target_root: target PURE_trans_row{i} root directory
        row_idx: row index to keep
        stmap_filename: name of STMap file (e.g., 'STMap.png')
    
    Returns:
        True if successful, False otherwise
    """
    source_subject = os.path.join(source_root, subject_name)
    target_subject = os.path.join(target_root, subject_name)
    
    if not os.path.isdir(source_subject):
        return False
    
    # Find STMap file
    source_stmap_dir = os.path.join(source_subject, 'STMap')
    stmap_path, found_name = find_stmap_file(source_stmap_dir)
    
    if stmap_path is None:
        print(f"  Warning: No STMap file found for {subject_name}")
        return False
    
    # Load original STMap
    original_stmap = cv2.imread(stmap_path)
    if original_stmap is None:
        print(f"  Warning: Could not load STMap for {subject_name}")
        return False
    
    # Verify shape: should be (25, T, 3)
    if original_stmap.shape[0] != 25:
        print(f"  Warning: {subject_name} STMap has {original_stmap.shape[0]} rows, expected 25")
        return False
    
    # Create modified STMap (keep only row_idx)
    modified_stmap = create_row_stmap(original_stmap, row_idx, keep_row=True)
    
    # Create target directory structure
    target_stmap_dir = os.path.join(target_subject, 'STMap')
    target_label_dir = os.path.join(target_subject, 'Label')
    os.makedirs(target_stmap_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)
    
    # Save modified STMap (use same filename as source)
    target_stmap_path = os.path.join(target_stmap_dir, stmap_filename)
    # Use PNG compression for PNG files, JPEG quality for JPEG files
    if stmap_filename.lower().endswith('.png'):
        cv2.imwrite(target_stmap_path, modified_stmap)
    else:
        cv2.imwrite(target_stmap_path, modified_stmap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    # Copy Label folder (contains BVP.mat, HR.mat, etc.)
    source_label_dir = os.path.join(source_subject, 'Label')
    if os.path.isdir(source_label_dir):
        # Copy all files from Label directory
        for item in os.listdir(source_label_dir):
            source_item = os.path.join(source_label_dir, item)
            target_item = os.path.join(target_label_dir, item)
            if os.path.isfile(source_item):
                shutil.copy2(source_item, target_item)
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, target_item, dirs_exist_ok=True)
    
    return True


def generate_row_dataset(row_idx, dry_run=False):
    """
    Generate modified STMap dataset for a specific row.
    
    Args:
        row_idx: row index (0-24)
        dry_run: if True, only print what would be done without creating files
    """
    target_root = f"{PURE_TRANS_BASE}{row_idx}"
    
    if not os.path.isdir(PURE_SOURCE_ROOT):
        print(f"Error: Source directory not found: {PURE_SOURCE_ROOT}")
        return False
    
    # Get list of subjects
    subjects = sorted([s for s in os.listdir(PURE_SOURCE_ROOT) 
                      if os.path.isdir(os.path.join(PURE_SOURCE_ROOT, s)) 
                      and not s.startswith('.')])
    
    if not subjects:
        print(f"Error: No subjects found in {PURE_SOURCE_ROOT}")
        return False
    
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Generating PURE_trans_row{row_idx} dataset")
    print(f"  Source: {PURE_SOURCE_ROOT}")
    print(f"  Target: {target_root}")
    print(f"  Subjects: {len(subjects)}")
    if row_idx == 0:
        print(f"  Row to keep: {row_idx} (zeroing rows 1-24)")
    elif row_idx == 24:
        print(f"  Row to keep: {row_idx} (zeroing rows 0-23)")
    else:
        print(f"  Row to keep: {row_idx} (zeroing rows 0-{row_idx-1} and {row_idx+1}-24)")
    
    if dry_run:
        print("\nWould process subjects:")
        for subj in subjects[:5]:
            print(f"  - {subj}")
        if len(subjects) > 5:
            print(f"  ... and {len(subjects) - 5} more")
        return True
    
    # Determine STMap filename from first subject
    first_subject = os.path.join(PURE_SOURCE_ROOT, subjects[0], 'STMap')
    _, stmap_filename = find_stmap_file(first_subject)
    if stmap_filename is None:
        print("Error: Could not determine STMap filename from source data")
        return False
    
    print(f"  STMap filename: {stmap_filename}\n")
    
    # Process each subject
    success_count = 0
    for subject in tqdm(subjects, desc=f"Processing row {row_idx}"):
        success = process_subject(subject, PURE_SOURCE_ROOT, target_root, row_idx, stmap_filename)
        if success:
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(subjects)} subjects processed successfully")
    print(f"Output directory: {target_root}")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description='Create modified PURE STMaps for row analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate row 0 only
  python create_pure_trans_stmaps.py --row 0
  
  # Generate all 25 rows
  python create_pure_trans_stmaps.py --all
  
  # Dry run (preview without creating files)
  python create_pure_trans_stmaps.py --row 0 --dry-run
        """
    )
    parser.add_argument('--row', type=int, choices=range(25),
                       help='Generate dataset for specific row (0-24)')
    parser.add_argument('--all', action='store_true',
                       help='Generate datasets for all 25 rows')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview actions without creating files')
    
    args = parser.parse_args()
    
    if args.all:
        # Generate all rows
        print("Generating datasets for all 25 rows...")
        for row_idx in range(25):
            generate_row_dataset(row_idx, dry_run=args.dry_run)
    elif args.row is not None:
        # Generate single row
        generate_row_dataset(args.row, dry_run=args.dry_run)
    else:
        parser.print_help()
        print("\nError: Must specify --row <0-24> or --all")


if __name__ == '__main__':
    main()
