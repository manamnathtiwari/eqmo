"""
Extract and process WESAD after manual browser download.
Run this AFTER you've downloaded WESAD.zip to data/wesad_raw/
"""
import os
import sys
import zipfile
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_pipeline.wesad_converter import process_wesad

def extract_and_process_wesad():
    """
    1. Extract WESAD.zip (if not already extracted)
    2. Process all subjects to HRV format
    """
    base_dir = Path(__file__).parent.parent.parent
    zip_path = base_dir / 'data' / 'wesad_raw' / 'WESAD.zip'
    extract_dir = base_dir / 'data'
    wesad_dir = base_dir / 'data' / 'WESAD'  # User extracted here
    output_dir = base_dir / 'data' / 'processed'
    
    # Step 1: Check if already extracted
    if wesad_dir.exists():
        print(f"✅ WESAD folder already exists at: {wesad_dir}")
    elif zip_path.exists():
        print(f"✅ Found WESAD.zip ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Step 2: Extract
        print(f"\nExtracting WESAD.zip...")
        print("This may take a few minutes...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total size
                total_size = sum(info.file_size for info in zip_ref.filelist)
                extracted_size = 0
                
                for info in zip_ref.filelist:
                    zip_ref.extract(info, extract_dir)
                    extracted_size += info.file_size
                    percent = (extracted_size / total_size) * 100
                    print(f"\rExtracting... {percent:.1f}%", end='')
                
                print(f"\n✅ Extraction complete!")
        except Exception as e:
            print(f"\n❌ Extraction failed: {e}")
            print("The zip file may be corrupt. Please re-download.")
            return False
    else:
        print(f"❌ WESAD data not found!")
        print(f"Expected either:")
        print(f"  - Extracted folder: {wesad_dir}")
        print(f"  - Zip file: {zip_path}")
        return False
    
    # Step 3: Verify WESAD directory exists
    if not wesad_dir.exists():
        print(f"❌ WESAD directory not found: {wesad_dir}")
        return False
    
    subjects = sorted([d for d in wesad_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
    
    if len(subjects) == 0:
        print(f"❌ No subject folders found in {wesad_dir}")
        return False
    
    print(f"\n📁 Found {len(subjects)} subjects: {[s.name for s in subjects]}")
    print(f"\nProcessing REAL WESAD data to HRV format...")
    print("=" * 70)
    
    success_count = 0
    failed_subjects = []
    
    for i, subject_dir in enumerate(subjects, 1):
        subject_id = subject_dir.name
        pkl_file = subject_dir / f'{subject_id}.pkl'
        
        if not pkl_file.exists():
            print(f"\n⚠️ [{i}/{len(subjects)}] {subject_id}: PKL file not found")
            failed_subjects.append(subject_id)
            continue
        
        try:
            print(f"\n[{i}/{len(subjects)}] Processing {subject_id}...", end='')
            
            # Map: S2 -> u_wesad_002, S3 -> u_wesad_003, etc.
            subject_num = int(subject_id[1:])
            output_id = f'u_wesad_{subject_num:03d}'
            
            # Process: ECG -> HRV (RMSSD) -> Stress
            process_wesad(
                input_path=str(pkl_file),
                output_dir=str(output_dir),
                subject_id=output_id
            )
            
            success_count += 1
            print(f" ✅ -> {output_id}.csv")
            
        except Exception as e:
            print(f" ❌ Failed: {e}")
            failed_subjects.append(subject_id)
    
    print("\n" + "=" * 70)
    print(f"\n🎉 Processing complete!")
    print(f"✅ Successfully processed: {success_count}/{len(subjects)} subjects")
    
    if failed_subjects:
        print(f"❌ Failed: {failed_subjects}")
    
    print(f"\n📁 Real data saved to: {output_dir}")
    print(f"\n🔬 Next steps:")
    print("1. Re-train model:      python src/models/train.py")
    print("2. SOTA comparison:     python src/evaluation/sota_comparison.py")
    print("3. Cohort simulation:   python src/evaluation/run_wesad_cohort.py")
    print("4. All evaluations:     python src/evaluation/ablation_study.py")
    
    return success_count > 0

if __name__ == "__main__":
    print("=" * 70)
    print("WESAD REAL DATA PROCESSOR")
    print("=" * 70)
    
    success = extract_and_process_wesad()
    
    if success:
        print("\n✅ ✅ ✅  READY TO USE REAL DATA!  ✅ ✅ ✅")
    else:
        print("\n❌ Processing failed. Check errors above.")
