import pandas as pd
import numpy as np
import pickle
import os
import argparse

def calculate_rmssd(ecg_signal, fs=700):
    """
    Calculates RMSSD (Root Mean Square of Successive Differences) from ECG signal.
    RMSSD is a time-domain measure of HRV, reflecting parasympathetic activity.
    Numpy-only implementation (no scipy).
    """
    rmssd, _ = calculate_rmssd_with_rr(ecg_signal, fs)
    return rmssd

def calculate_rmssd_with_rr(ecg_signal, fs=700):
    """
    Calculates RMSSD and returns RR intervals for heart rate calculation.
    Returns: (rmssd, rr_intervals)
    """
    # 1. Peak Detection (R-peaks)
    # Simple threshold-based detection
    
    # Normalize signal
    ecg_mean = np.mean(ecg_signal)
    ecg_std = np.std(ecg_signal)
    if ecg_std == 0:
        return np.nan, None
        
    ecg_norm = (ecg_signal - ecg_mean) / ecg_std
    
    # Threshold
    threshold = 3.0
    
    # Find indices where signal > threshold
    above_threshold = ecg_norm > threshold
    
    # Find rising edges (transition from False to True)
    # This gives us the start of each peak region
    # We want the maximum within each region
    
    if not np.any(above_threshold):
        return np.nan, None
        
    # Simple approach: Find local maxima
    # A point is a peak if it's larger than its neighbors
    # and above threshold
    
    # Shifted arrays
    center = ecg_norm[1:-1]
    left = ecg_norm[:-2]
    right = ecg_norm[2:]
    
    # Identify peaks
    is_peak = (center > left) & (center > right) & (center > threshold)
    peak_indices = np.where(is_peak)[0] + 1 # Adjust for shift
    
    if len(peak_indices) < 2:
        return np.nan, None
        
    # Refractory period check (distance)
    # 0.4s * fs
    min_distance = int(0.4 * fs)
    
    cleaned_peaks = []
    last_peak = -min_distance
    
    # Greedy selection (assuming peaks are sorted by time)
    # For R-peaks, usually the highest one in a window is the R-peak.
    # But simple distance check often works for clean data.
    
    # Better: Sort by amplitude? No, time order matters for HRV.
    # We iterate through time.
    
    for p in peak_indices:
        if p - last_peak > min_distance:
            cleaned_peaks.append(p)
            last_peak = p
        else:
            # If this peak is higher than the last one, replace it?
            # (Simple R-peak correction)
            if ecg_norm[p] > ecg_norm[last_peak]:
                cleaned_peaks[-1] = p
                last_peak = p
                
    if len(cleaned_peaks) < 2:
        return np.nan, None
        
    # 2. Calculate RR intervals (in milliseconds)
    peaks = np.array(cleaned_peaks)
    rr_intervals = np.diff(peaks) / fs * 1000  # ms
    
    # 3. Filter artifacts (simple range check: 300ms to 2000ms)
    valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
    
    if len(valid_rr) < 2:
        return np.nan, None
        
    # 4. Calculate Successive Differences
    diff_rr = np.diff(valid_rr)
    
    # 5. Calculate RMSSD
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    return rmssd, valid_rr


def process_wesad(input_path, output_dir, subject_id):
    """
    Reads a WESAD pickle file and converts it to our project's CSV format.
    ENHANCED: Multi-modal feature extraction from ALL available sensors.
    
    Features extracted:
    - ECG: HRV (RMSSD, SDNN, pNN50, LF/HF ratio), Heart Rate
    - EDA: Mean, Std, Peak count
    - Respiration: Rate, depth variability
    - Temperature: Mean, variability
    - Accelerometer: Activity level (magnitude)
    - EMG: Mean activity level
    """
    print(f"Processing {input_path}...")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    # Extract ALL signals from chest sensor
    chest_data = data['signal']['chest']
    ecg_signal = chest_data['ECG'].flatten()
    eda_signal = chest_data['EDA'].flatten()
    temp_signal = chest_data['Temp'].flatten()
    resp_signal = chest_data['Resp'].flatten()
    acc_signal = chest_data['ACC']  # 3-axis: (n_samples, 3)
    emg_signal = chest_data['EMG'].flatten()
    labels = data['label'].flatten()
    
    # WESAD Sampling Rate = 700Hz
    fs = 700
    
    # Calculate features in sliding windows (60 seconds)
    window_sec = 60
    step_sec = 1  # 1 second steps for high resolution
    window_samples = window_sec * fs
    step_samples = step_sec * fs
    
    n_samples = len(ecg_signal)
    
    results = []
    
    print("Extracting multi-modal features (ECG, EDA, Temp, Resp, ACC, EMG)...")
    print("This may take several minutes...")
    
    for i in range(0, n_samples - window_samples, step_samples):
        # Extract windows for all signals
        window_ecg = ecg_signal[i : i + window_samples]
        window_eda = eda_signal[i : i + window_samples]
        window_temp = temp_signal[i : i + window_samples]
        window_resp = resp_signal[i : i + window_samples]
        window_acc = acc_signal[i : i + window_samples, :]
        window_emg = emg_signal[i : i + window_samples]
        window_label = labels[i : i + window_samples]
        
        # ==========================================
        # 1. ECG FEATURES (HRV + Heart Rate)
        # ==========================================
        rmssd, rr_intervals = calculate_rmssd_with_rr(window_ecg, fs)
        
        # Heart Rate
        if rr_intervals is not None and len(rr_intervals) > 0:
            mean_rr = np.mean(rr_intervals)
            heart_rate = 60000.0 / mean_rr if mean_rr > 0 else np.nan
            
            # Additional HRV time-domain features
            sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else np.nan
            
            # pNN50: percentage of successive RR intervals differing by >50ms
            if len(rr_intervals) > 1:
                diff_rr = np.abs(np.diff(rr_intervals))
                pnn50 = (np.sum(diff_rr > 50) / len(diff_rr)) * 100
            else:
                pnn50 = np.nan
                
            # Frequency-domain HRV (LF/HF ratio)
            try:
                if len(rr_intervals) > 10:  # Need sufficient data for FFT
                    # Resample to 4 Hz for frequency analysis
                    from scipy.interpolate import interp1d
                    from scipy.signal import welch
                    
                    time_rr = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
                    time_uniform = np.arange(0, time_rr[-1], 0.25)  # 4 Hz
                    
                    # Interpolate
                    f_interp = interp1d(time_rr, rr_intervals, kind='cubic', fill_value='extrapolate')
                    rr_uniform = f_interp(time_uniform)
                    
                    # Compute PSD
                    freqs, psd = welch(rr_uniform, fs=4, nperseg=min(256, len(rr_uniform)))
                    
                    # Extract frequency bands
                    lf_band = (freqs >= 0.04) & (freqs < 0.15)
                    hf_band = (freqs >= 0.15) & (freqs < 0.4)
                    
                    lf_power = np.trapz(psd[lf_band], freqs[lf_band])
                    hf_power = np.trapz(psd[hf_band], freqs[hf_band])
                    
                    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
                else:
                    lf_hf_ratio = np.nan
            except:
                lf_hf_ratio = np.nan
        else:
            heart_rate = np.nan
            sdnn = np.nan
            pnn50 = np.nan
            lf_hf_ratio = np.nan
        
        # ==========================================
        # 2. EDA FEATURES (Electrodermal Activity)
        # ==========================================
        eda_mean = np.mean(window_eda)
        eda_std = np.std(window_eda)
        
        # EDA peak count (Skin Conductance Responses - SCRs)
        # Simple peak detection: derivative method
        eda_diff = np.diff(window_eda)
        # Peaks: where derivative changes from positive to negative
        eda_peaks = np.sum((eda_diff[:-1] > 0) & (eda_diff[1:] < 0))
        
        # ==========================================
        # 3. RESPIRATION FEATURES
        # ==========================================
        resp_mean = np.mean(window_resp)
        resp_std = np.std(window_resp)
        
        # Respiration rate: count zero-crossings
        resp_centered = window_resp - np.mean(window_resp)
        zero_crossings = np.sum((resp_centered[:-1] * resp_centered[1:]) < 0)
        resp_rate = (zero_crossings / 2) / window_sec * 60  # breaths per minute
        
        # ==========================================
        # 4. TEMPERATURE FEATURES
        # ==========================================
        temp_mean = np.mean(window_temp)
        temp_std = np.std(window_temp)
        
        # ==========================================
        # 5. ACCELEROMETER FEATURES (Activity Level)
        # ==========================================
        # Compute magnitude: sqrt(x^2 + y^2 + z^2)
        acc_magnitude = np.sqrt(np.sum(window_acc**2, axis=1))
        activity_level = np.mean(acc_magnitude)
        activity_std = np.std(acc_magnitude)
        
        # ==========================================
        # 6. EMG FEATURES (Muscle Activity)
        # ==========================================
        emg_mean = np.mean(np.abs(window_emg))  # Rectified EMG
        emg_std = np.std(window_emg)
        
        # ==========================================
        # 7. LABEL (Ground Truth)
        # ==========================================
        label_mode = float(np.median(window_label))
        
        # Time in minutes
        time_min = (i + window_samples/2) / fs / 60.0
        
        # Collect all features
        results.append({
            'time': time_min,
            
            # ECG-derived (HRV + HR)
            'hrv_rmssd': rmssd,
            'hrv_sdnn': sdnn,
            'hrv_pnn50': pnn50,
            'hrv_lf_hf': lf_hf_ratio,
            'heart_rate': heart_rate,
            
            # EDA features
            'eda_mean': eda_mean,
            'eda_std': eda_std,
            'eda_peaks': eda_peaks,
            
            # Respiration features
            'resp_mean': resp_mean,
            'resp_std': resp_std,
            'resp_rate': resp_rate,
            
            # Temperature features
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            
            # Accelerometer features
            'activity_level': activity_level,
            'activity_std': activity_std,
            
            # EMG features
            'emg_mean': emg_mean,
            'emg_std': emg_std,
            
            # Label
            'label': label_mode
        })
        
        # Progress indicator
        if (i // step_samples) % 100 == 0:
            progress = (i / (n_samples - window_samples)) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')
        
    print(f"\n  Extracted {len(results)} feature windows")
    
    df = pd.DataFrame(results)
    
    # Handle NaNs (if any windows failed feature calculation)
    # Using modern pandas syntax (fillna with method= is deprecated)
    df = df.interpolate(method='linear').bfill().ffill()
    
    # Normalize Heart Rate to [0, 1] as workload proxy
    hr_min = df['heart_rate'].min()
    hr_max = df['heart_rate'].max()
    df['workload'] = (df['heart_rate'] - hr_min) / (hr_max - hr_min + 1e-6)
    
    # Derive stress from labels (will be overridden during training normalization)
    # This is just for initial visualization
    label_to_stress = {1: 0.2, 2: 1.0, 3: 0.3, 4: 0.1, 0: 0.2}
    df['stress'] = df['label'].map(lambda x: label_to_stress.get(int(x), 0.2))
    
    # Save
    out_file = os.path.join(output_dir, f"{subject_id}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df.columns)} features to {out_file}")
    print(f"Features: {list(df.columns)}")
    
    return df

def generate_mock_wesad(output_path):
    """
    Generates a REALISTIC mock WESAD-like pickle file for testing the pipeline.
    Simulates ECG with variable Heart Rate and HRV based on stress labels.
    """
    print(f"Generating REALISTIC mock WESAD data at {output_path}...")
    
    fs = 700
    duration_sec = 3600 # 1 hour
    n_samples = fs * duration_sec
    
    # Labels: 0-20m Baseline (1), 20-40m Stress (2), 40-60m Recovery (3)
    # WESAD uses 1=Baseline, 2=Stress, 3=Amusement, 4=Meditation
    labels = np.ones(n_samples) # Default Baseline
    labels[20*60*fs : 40*60*fs] = 2 # Stress
    labels[40*60*fs : ] = 3 # Recovery
    
    # Generate RR intervals
    # Baseline: 60-70 bpm (1.0 - 0.85s interval), High HRV (std=0.08s)
    # Stress: 90-110 bpm (0.66 - 0.55s interval), Low HRV (std=0.01s)
    
    rr_intervals = []
    current_time = 0
    
    while current_time < duration_sec:
        # Determine state
        idx = int(current_time * fs)
        if idx >= n_samples: break
        label = labels[idx]
        
        if label == 2: # Stress
            mean_rr = 0.6 # 100 bpm
            std_rr = 0.015 # Low variability
        else: # Baseline/Recovery
            mean_rr = 0.9 # 67 bpm
            std_rr = 0.08 # High variability (RSA)
            
        # Sample RR
        rr = np.random.normal(mean_rr, std_rr)
        rr = max(0.3, min(2.0, rr)) # Clip
        
        rr_intervals.append(rr)
        current_time += rr
        
    # Construct ECG from RR
    ecg = np.zeros(n_samples)
    peak_times = np.cumsum(rr_intervals)
    peak_indices = (peak_times * fs).astype(int)
    peak_indices = peak_indices[peak_indices < n_samples]
    
    # Add QRS complexes
    # Vectorized placement would be faster but loop is fine for 1 hour (~4000 beats)
    for p in peak_indices:
        # Simple QRS shape
        width = int(0.04 * fs) # 40ms half-width
        start = max(0, p - width)
        end = min(n_samples, p + width)
        
        if start >= end: continue
        
        # R-wave shape (gaussian-ish)
        # Map indices to range -3 to 3
        x = np.linspace(-3, 3, end-start)
        shape = 5.0 * np.exp(-x**2)
        ecg[start:end] += shape
        
    # Add noise
    ecg += np.random.normal(0, 0.1, n_samples)
    
    data = {
        'signal': {
            'chest': {
                'ECG': ecg.reshape(-1, 1),
                'EDA': np.random.normal(10, 2, (n_samples, 1)), # Legacy
                'Temp': np.random.normal(30, 1, (n_samples, 1))
            }
        },
        'label': labels
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print("Mock generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mock', 'process'], required=True)
    parser.add_argument('--input', help='Path to input pickle file')
    parser.add_argument('--output', help='Path to output file or directory')
    parser.add_argument('--subject', help='Subject ID (e.g., u_wesad_01)', default='u_wesad_01')
    
    args = parser.parse_args()
    
    if args.mode == 'mock':
        generate_mock_wesad(args.output)
    elif args.mode == 'process':
        process_wesad(args.input, args.output, args.subject)
