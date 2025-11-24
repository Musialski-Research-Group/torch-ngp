import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- Configuration ---
# Set the root directory containing the 'drums', 'ficus', etc. folders
PARENT_LOG_DIR = './results/grouped_runs'

# The subdirectory path relative to the experiment folder where the tfevents files are located.
RUN_SUBPATH = 'run/ngp'

# The TensorBoard tag you want to plot (e.g., 'train/loss')
TARGET_TAG = 'train/loss'

# --- Visualization Options ---

# Targeting specific heavy smoothing levels (alpha value)
SMOOTHING_OPTIONS = {
    "HeavySmooth": 0.1,
    "ExtraHeavySmooth": 0.05,
    "UltraSmooth": 0.02 
}

# Y-Axis Scale is fixed to linear as requested
Y_SCALE_OPTIONS = ["linear"] 

# Define color maps for variation
# FIX: The key must be the lowercase name 'tab10' to be a valid colormap name.
COLOR_SCHEMES = {
    "tab10": 'tab10', # <-- CORRECTED KEY
    "Set1": 'Set1',
    "Dark2": 'Dark2',
    "Paired": 'Paired'
}

# Define the output directory for all generated plots
OUTPUT_DIR = 'comparison_plots_focused_smooth'

# --- Plotting Constraints ---
Y_AXIS_TOP_LIMIT = 0.04 

# --- Helper Functions (Omitted for brevity, they are unchanged) ---

def get_run_log_dirs(group_dir):
    """Finds the full path to every TensorBoard log directory."""
    log_dirs = []
    for example_name in os.listdir(group_dir):
        full_path = os.path.join(group_dir, example_name)
        run_path = os.path.join(full_path, RUN_SUBPATH)
        if os.path.isdir(full_path) and os.path.isdir(run_path):
            log_dirs.append(run_path)
    return log_dirs

def load_scalar_data(log_dir, tag):
    """Loads scalar data for a single run from the specified directory."""
    try:
        event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
        event_acc.Reload()
        
        if tag not in event_acc.Tags().get('scalars', []):
            return None
            
        scalars = event_acc.Scalars(tag)
        data = pd.DataFrame([(s.step, s.value) for s in scalars], columns=['step', 'value'])
        return data
        
    except Exception as e:
        print(f"Error reading log file in {log_dir}: {e}")
        return None

def process_single_group(group_path, tag):
    """
    Processes all runs within a single group directory, calculates the mean and std,
    and returns the resulting DataFrame for plotting.
    """
    all_run_log_dirs = get_run_log_dirs(group_path)
    all_run_data = [] 
    
    if not all_run_log_dirs:
        return None, 0

    # 1. Load data from every single run in this group
    for log_dir in all_run_log_dirs:
        run_data_df = load_scalar_data(log_dir, tag)
        
        if run_data_df is not None and not run_data_df.empty:
            # FIX 1: Deduplicate steps
            run_data_df = run_data_df.drop_duplicates(subset=['step'], keep='last')
            all_run_data.append(run_data_df)

    if not all_run_data:
        return None, 0

    # 2. Alignment and Grand Averaging for this group
    common_steps = pd.Index([])
    for df in all_run_data:
         common_steps = common_steps.union(df['step'])
    common_steps = common_steps.sort_values()

    reindexed_values = []
    for df in all_run_data:
        # Set 'step' as index, then reindex and linearly interpolate
        df_reindexed = df.set_index('step')['value'].reindex(common_steps).interpolate(method='linear')
        
        # FIX 2: Fill trailing NaN values
        df_reindexed = df_reindexed.fillna(method='ffill') 
        
        reindexed_values.append(df_reindexed.astype(float))

    # Calculate the mean and standard deviation
    combined_df = pd.concat(reindexed_values, axis=1)
    mean_values = combined_df.mean(axis=1)
    std_values = combined_df.std(axis=1)
    
    # Final Guard: Ensure float type for Matplotlib
    mean_values = pd.Series(mean_values.values.astype(np.float64), index=mean_values.index)
    std_values = pd.Series(std_values.values.astype(np.float64), index=std_values.index)

    # Package the results
    result_df = pd.DataFrame({
        'step': common_steps.astype(np.float64), 
        'mean': mean_values, 
        'std': std_values
    })
    
    return result_df, len(all_run_data)


# --- Plotting Function ---

def plot_and_save(all_group_results, alpha, y_scale, color_scheme_name):
    """Generates and saves a single plot based on the given parameters."""
    
    smooth_label = [k for k, v in SMOOTHING_OPTIONS.items() if v == alpha][0]
    
    plt.figure(figsize=(12, 8))
    
    # Get the color map
    # We now pass the key (e.g., 'tab10') directly to get_cmap.
    color_map = plt.cm.get_cmap(color_scheme_name).colors
    
    # 1. Apply Smoothing to the Mean Curve
    smoothed_results = {}
    for group_name, data_dict in all_group_results.items():
        data = data_dict['data'].copy()
        
        if alpha < 1.0:
            # Apply EWMA smoothing directly to the mean values
            data['mean_smoothed'] = data['mean'].ewm(alpha=alpha, adjust=False).mean()
        else:
            data['mean_smoothed'] = data['mean']
            
        smoothed_results[group_name] = data
        
    # 2. Plotting Loop
    for i, (group_name, data) in enumerate(smoothed_results.items()):
        
        steps = data['step']
        mean_smoothed = data['mean_smoothed']
        std = data['std']
        
        # Determine color using the color_map
        color = color_map[i % len(color_map)]

        # Plot the standard deviation as a DIMMED shaded area (variance)
        plt.fill_between(
            steps, 
            data['mean'] - std, # Use UNSMOOTHED mean for variance calculation bounds
            data['mean'] + std, 
            color=color, 
            alpha=0.15, 
            label='_nolegend_' 
        )
        
        # Plot the mean curve (vivid)
        plt.plot(
            steps, 
            mean_smoothed, 
            color=color, 
            linewidth=2.5, 
            label=f'{group_name}' # Legend only includes the group name
        )

    # 3. Finalize Plot Details
    plt.yscale(y_scale)
    
    # Apply Y-axis limit (only for linear scale)
    if y_scale == 'linear':
        plt.ylim(bottom=0, top=Y_AXIS_TOP_LIMIT)

    # Title is removed. Using text box for descriptive metadata.
    # plt.text(0.5, 1.01, 
    #          f'Y-Scale: {y_scale.title()} | Smoothing: {smooth_label} | Color Palette: {color_scheme_name}', 
    #          transform=plt.gca().transAxes, 
    #          fontsize=12, ha='center')

    plt.xlabel('Step')
    plt.ylabel(TARGET_TAG.split('/')[-1].replace('_', ' ').title())
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Legend only has the run names (no title)
    plt.legend(loc='upper right', title=None)
    
    # 4. Save the figure
    filename = f"plot_{TARGET_TAG.replace('/', '_')}_Y_{y_scale}_S_{smooth_label}_C_{color_scheme_name}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory
    print(f"  - Saved: {save_path}")


# --- Main Execution ---

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    group_dirs = [d for d in os.listdir(PARENT_LOG_DIR) if os.path.isdir(os.path.join(PARENT_LOG_DIR, d))]
    
    if not group_dirs:
        print(f"Error: No group directories found in {PARENT_LOG_DIR}. Check your path.")
        return

    all_group_results = {}
    print(f"Found {len(group_dirs)} experimental groups to process.")
    
    # 1. Process all groups once (data loading and averaging)
    for group_name in group_dirs:
        group_path = os.path.join(PARENT_LOG_DIR, group_name)
        print(f"Loading and Averaging Group: **{group_name}**")
        
        result_df, num_runs = process_single_group(group_path, TARGET_TAG)
        
        if result_df is not None:
            all_group_results[group_name] = {'data': result_df, 'runs': num_runs}

    if not all_group_results:
        print("\nNo valid data found in any group to plot.")
        return

    # 2. Generate all plot variations
    print("\nStarting batch plot generation for all combinations...")
    
    for color_scheme_name in COLOR_SCHEMES.keys(): # Keys are now 'tab10', 'Set1', etc.
        for y_scale in Y_SCALE_OPTIONS: 
            for smooth_name, alpha in SMOOTHING_OPTIONS.items():
                plot_and_save(all_group_results, alpha, y_scale, color_scheme_name)

    print("\nBatch plotting complete! All plots saved to the output directory.")


if __name__ == '__main__':
    main()