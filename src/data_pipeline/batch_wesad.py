import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_pipeline.wesad_converter import generate_mock_wesad, process_wesad

def generate_all_subjects():
    """Generate mock WESAD data for 15 subjects (S2-S16)"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'wesad_sample')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating 15 mock WESAD subjects...")
    for i in range(2, 17):  # S2 to S16
        subject_file = os.path.join(output_dir, f'S{i}.pkl')
        # Force regeneration to use new HRV logic
        print(f"  Generating S{i}...")
        generate_mock_wesad(subject_file)
    
    print("Done generating subjects.")

def process_all_subjects():
    """Process all 15 WESAD subjects to CSV format"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_dir = os.path.join(base_dir, 'data', 'wesad_sample')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing 15 WESAD subjects...")
    for i in range(2, 17):  # S2 to S16
        subject_id = f"u_wesad_{i:03d}"
        input_file = os.path.join(input_dir, f'S{i}.pkl')
        
        if os.path.exists(input_file):
            print(f"  Processing S{i} -> {subject_id}...")
            process_wesad(input_file, output_dir, subject_id)
        else:
            print(f"  Warning: {input_file} not found, skipping.")
    
    print("Done processing subjects.")

if __name__ == "__main__":
    generate_all_subjects()
    process_all_subjects()
