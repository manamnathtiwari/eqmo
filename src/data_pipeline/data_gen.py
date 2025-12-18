import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def simulate_user(user_id, n_steps=10080, dt=1.0):
    """
    Simulate a single user's stress trajectory over time.
    n_steps: 10080 minutes = 7 days (if dt=1 min)
    """
    # Parameters for this user
    # Recovery rate (beta): 0.01 to 0.05
    beta = np.random.uniform(0.01, 0.05)
    # Sensitivity to workload (alpha): 0.05 to 0.2
    alpha = np.random.uniform(0.05, 0.2)
    # Baseline stress level
    baseline = np.random.uniform(0.1, 0.3)
    
    # Initial state
    S = baseline
    
    # Generate synthetic workload (e.g., spikes during day, low at night)
    # Simple model: sinusoidal pattern + random events
    time = np.arange(n_steps) * dt
    # Day/night cycle (period = 1440 mins)
    circadian = 0.5 * (1 - np.cos(2 * np.pi * time / 1440))
    # Random workload spikes
    workload_events = np.random.exponential(scale=100, size=n_steps)
    workload_events = (workload_events > 400).astype(float) * np.random.uniform(0.5, 1.0, size=n_steps)
    
    workload = 0.3 * circadian + 0.7 * workload_events
    
    # Simulation loop
    stress_trajectory = []
    workload_trajectory = []
    
    for t in range(n_steps):
        w = workload[t]
        # Dynamics: dS = (alpha * w - beta * (S - baseline)) * dt + noise
        noise = np.random.normal(0, 0.01)
        dS = (alpha * w - beta * (S - baseline)) * dt + noise
        S = S + dS
        
        # Clip to reasonable range [0, 1]
        S = np.clip(S, 0, 1)
        
        stress_trajectory.append(S)
        workload_trajectory.append(w)
        
    df = pd.DataFrame({
        'time': time,
        'stress': stress_trajectory,
        'workload': workload_trajectory,
        'user_id': user_id,
        'alpha': alpha,
        'beta': beta,
        'baseline': baseline
    })
    
    return df

def main():
    output_dir = os.path.join('data', 'synthetic')
    os.makedirs(output_dir, exist_ok=True)
    
    n_users = 50
    print(f"Generating data for {n_users} users...")
    
    all_metadata = []
    
    for i in tqdm(range(n_users)):
        user_id = f"u_{i:03d}"
        df = simulate_user(user_id)
        
        # Save individual file
        df.to_csv(os.path.join(output_dir, f"{user_id}.csv"), index=False)
        
        # Collect metadata
        meta = df.iloc[0][['user_id', 'alpha', 'beta', 'baseline']].to_dict()
        all_metadata.append(meta)
        
    # Save metadata
    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print("Done! Data saved to", output_dir)

if __name__ == "__main__":
    main()
