"""
Download WESAD dataset with resume capability.
Handles connection interruptions and large files.
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_wesad_robust():
    """
    Downloads WESAD dataset with resume support.
    """
    base_dir = Path(__file__).parent.parent.parent
    download_dir = base_dir / 'data' / 'wesad_raw'
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Original WESAD download link
    url = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
    zip_path = download_dir / 'WESAD.zip'
    
    # Check if partial download exists
    resume_header = {}
    if zip_path.exists():
        downloaded_size = zip_path.stat().st_size
        resume_header = {'Range': f'bytes={downloaded_size}-'}
        print(f"Resuming download from {downloaded_size / 1024 / 1024:.1f} MB...")
    else:
        downloaded_size = 0
        print("Starting fresh download...")
    
    print(f"Downloading WESAD dataset to: {zip_path}")
    print("This is a large file (~1.8 GB), please be patient...")
    
    try:
        response = requests.get(url, headers=resume_header, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0)) + downloaded_size
        
        mode = 'ab' if downloaded_size > 0 else 'wb'
        
        with open(zip_path, mode) as f, tqdm(
            total=total_size,
            initial=downloaded_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="WESAD Download"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"\n✅ Download complete: {zip_path}")
        print(f"File size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Extract
        print("\nExtracting files...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        print("✅ Extraction complete!")
        
        # List subjects
        wesad_folder = download_dir / 'WESAD'
        if wesad_folder.exists():
            subjects = sorted([d.name for d in wesad_folder.iterdir() if d.is_dir()])
            print(f"\n📁 Found {len(subjects)} subjects: {subjects}")
            return True
        else:
            print("⚠️ WESAD folder not found after extraction")
            return False
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"Partial download saved. Run again to resume.")
        return False

if __name__ == "__main__":
    success = download_wesad_robust()
    if success:
        print("\n🎉 WESAD dataset ready!")
        print("Next step: python src/data_pipeline/batch_wesad.py --real")
    else:
        print("\n⚠️ Please try again when connection is stable")
