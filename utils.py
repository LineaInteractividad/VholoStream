# utils.py
import os
import pandas as pd

def datasets_reading(experiment_fps, experiment_number_of_points, alone_dir, together_dir, numeric_cols):
    """
    Reads all datasets (Huawei alone, Laptop alone, Huawei together, and Laptop together),
    cleans them, and returns dictionaries of DataFrames keyed by "<fps>_<points>".
    """
    
    # Dictionaries to store the DataFrames
    huawei_alone_dataframes = {}
    laptop_alone_dataframes = {}
    huawei_together_dataframes = {}
    laptop_together_dataframes = {}
    
    # Helper function to read and clean a single CSV file
    def read_and_clean_csv(filepath):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath,dtype=str, na_values=['nan', 'undefined'])
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Convert specified columns to numeric
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows that have all NaN in numeric columns
            numeric_cols_in_df = [col for col in numeric_cols if col in df.columns]
            if numeric_cols_in_df:
                df.dropna(subset=numeric_cols_in_df, how='all', inplace=True)
            return df
        return None

    for fps in experiment_fps:
        for points in experiment_number_of_points:
            # Filenames based on the naming convention
            huawei_alone_filename = f"HuaweiNova6-{fps}fps-{points}K-Huawei Nova6.csv"
            laptop_alone_filename = f"Laptop-{fps}fps-{points}K-Laptop.csv"

            huawei_together_filename = f"HuaweiNova6-{fps}fps-{points}K-Huawei Nova6.csv"
            laptop_together_filename = f"Laptop-{fps}fps-{points}K-Laptop.csv"

            # Paths
            huawei_alone_path = os.path.join(alone_dir, huawei_alone_filename)
            laptop_alone_path = os.path.join(alone_dir, laptop_alone_filename)

            huawei_together_path = os.path.join(together_dir, huawei_together_filename)
            laptop_together_path = os.path.join(together_dir, laptop_together_filename)

            # Key for the dicts
            key = f"{fps}_{points}"

            # Read Huawei Alone
            huawei_df = read_and_clean_csv(huawei_alone_path)
            if huawei_df is not None:
                huawei_alone_dataframes[key] = huawei_df

            # Read Laptop Alone
            laptop_df = read_and_clean_csv(laptop_alone_path)
            if laptop_df is not None:
                laptop_alone_dataframes[key] = laptop_df

            # Read Huawei Concurrent
            huawei_t_df = read_and_clean_csv(huawei_together_path)
            if huawei_t_df is not None:
                huawei_together_dataframes[key] = huawei_t_df

            # Read Laptop Concurrent
            laptop_t_df = read_and_clean_csv(laptop_together_path)
            if laptop_t_df is not None:
                laptop_together_dataframes[key] = laptop_t_df

    return huawei_alone_dataframes, laptop_alone_dataframes, huawei_together_dataframes, laptop_together_dataframes


def add_derived_metrics(df, fps_points_key):
    """
    Adds derived columns to the DataFrame that represent various time intervals 
    in the player pipeline, as well as the instantaneous FPS and a moving average of that FPS.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the player metrics.
    fps_points_key : str
        The key from the dictionary, expected to be in the format "<fps>_<points>",
        used to extract the nominal FPS.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with added derived metric columns.
    """
    # Parse nominal FPS from the key
    # Expected format: "<fps>_<points>"
    try:
        fps_str, _ = fps_points_key.split('_')
        nominal_fps = int(fps_str)
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse FPS from key '{fps_points_key}'. Expected format '<fps>_<points>'.")

    # Ensure columns are stripped of whitespace
    df.columns = df.columns.str.strip()

    # Derived metrics
    if all(col in df.columns for col in ["Bf1 In", "Bf1 out"]):
        df["Buffering 1 Time"] = df["Bf1 out"] - df["Bf1 In"]
    if all(col in df.columns for col in ["Bf2 In", "Bf2 out"]):
        df["Buffering 2 Time"] = df["Bf2 out"] - df["Bf2 In"]
    if all(col in df.columns for col in ["RC Timestamp", "DC Timestamp"]):
        df["DC Time"] = df["DC Timestamp"] - df["RC Timestamp"]
    if all(col in df.columns for col in ["RC Timestamp", "PL Timestamp"]):
        df["Processing Time"] = df["PL Timestamp"] - df["RC Timestamp"]

    # Compute fps_current
    # fps_current for the first row = nominal_fps
    # fps_current for subsequent rows = 1000 / (PL_current - PL_previous)
    if "PL Timestamp" in df.columns:
        df["fps_current"] = None
        df.iloc[0, df.columns.get_loc("fps_current")] = nominal_fps
        for i in range(1, len(df)):
            pl_diff = df["PL Timestamp"].iloc[i] - df["PL Timestamp"].iloc[i-1]
            if pl_diff != 0:
                df.iat[i, df.columns.get_loc("fps_current")] = 1000.0 / pl_diff
            else:
                df.iat[i, df.columns.get_loc("fps_current")] = nominal_fps
    else:
        # If PL Timestamp doesn't exist, we can't compute fps_current
        df["fps_current"] = nominal_fps

    # Compute fps_moving_media as a rolling mean of fps_current over a window of 10
    # Note: Converting fps_current to a numeric dtype before rolling mean
    df["fps_current"] = pd.to_numeric(df["fps_current"], errors='coerce')
    df["fps_moving_media"] = df["fps_current"].rolling(window=10, min_periods=1).mean()

    return df


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_experiment_comparison(
    fps,
    y_min,
    y_max,
    column,
    experiment_number_of_points,
    alone_dataframes,
    together_dataframes,
    device_name="Device"
):
    """
    Plots a comparison between a device alone and the same device together with another scenario
    for a given FPS, focusing on a specified numeric column. It produces two subplots:
    one for the full range and one zoomed in between y_min and y_max.

    Parameters
    ----------
    fps : int
        The frame rate of the experiments to filter data by.
    y_min : float
        The lower bound of the y-axis for the zoomed-in subplot.
    y_max : float
        The upper bound of the y-axis for the zoomed-in subplot.
    column : str
        The name of the column in the dataframes to plot on the y-axis.
    experiment_number_of_points : list
        List of point cloud sizes used in the experiments.
    alone_dataframes : dict
        Dictionary of DataFrames keyed by "<fps>_<points>" for the device alone scenario.
    together_dataframes : dict
        Dictionary of DataFrames keyed by "<fps>_<points>" for the device together scenario.
    device_name : str, optional
        The name of the device (e.g., "Huawei", "Laptop") to display in titles and scenario labels.
        Default is "Device".
    """

    scenario_alone = f"{device_name} Alone"
    scenario_together = f"{device_name} Together"

    data_list = []

    # Gather data from both scenarios at the given fps
    for points in experiment_number_of_points:
        key = f"{fps}_{points}"

        # Alone scenario
        if key in alone_dataframes:
            df_alone = alone_dataframes[key].copy()
            df_alone["scenario"] = scenario_alone
            df_alone["points"] = points
            data_list.append(df_alone)

        # Together scenario
        if key in together_dataframes:
            df_together = together_dataframes[key].copy()
            df_together["scenario"] = scenario_together
            df_together["points"] = points
            data_list.append(df_together)

    # Concatenate all the data for plotting
    if not data_list:
        print("No data found for the given parameters.")
        return

    combined_df = pd.concat(data_list, ignore_index=True)

    sns.set_style("whitegrid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # First subplot: full range
    sns.lineplot(
        data=combined_df,
        x="Frame Number",
        y=column,
        hue="points",
        style="scenario",
        ax=ax1
    )
    ax1.set_title(f"Full Range of {column} at {fps} FPS for {device_name}")
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel(f"{column} (ms)")
    ax1.legend(title="Points / Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Second subplot: zoomed in range
    sns.lineplot(
        data=combined_df,
        x="Frame Number",
        y=column,
        hue="points",
        style="scenario",
        ax=ax2,
        legend=False  # We'll rely on the legend in the first subplot
    )
    ax2.set_ylim(y_min, y_max)
    ax2.set_title(f"Zoomed Range ({y_min} to {y_max} ms) for {device_name}")
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel(f"{column} (ms)")

    plt.tight_layout()
    plt.show()

def plot_experiment_fps(
    fps,
    y_min=0,
    y_max=None,
    experiment_number_of_points=None,
    alone_dataframes=None,
    together_dataframes=None,
    device_name="Device"
):
    """
    Plots a comparison of fps_current and fps_moving_media for a device tested alone and together,
    at a given FPS. Two subplots are created:
    1. fps_current vs. Frame Number
    2. fps_moving_media vs. Frame Number

    Parameters
    ----------
    fps : int
        The frame rate of the experiments to filter data by.
    y_min : float
        The lower bound of the y-axis. Default is 0.
    y_max : float or None
        The upper bound of the y-axis. If None, the y-axis will auto-adjust.
    experiment_number_of_points : list
        List of point cloud sizes used in the experiments.
    alone_dataframes : dict
        Dictionary of DataFrames keyed by "<fps>_<points>" for the device alone scenario.
    together_dataframes : dict
        Dictionary of DataFrames keyed by "<fps>_<points>" for the device together scenario.
    device_name : str, optional
        The name of the device (e.g., "Huawei", "Laptop") to display in titles and scenario labels.
        Default is "Device".
    """

    if experiment_number_of_points is None or alone_dataframes is None or together_dataframes is None:
        raise ValueError("experiment_number_of_points, alone_dataframes, and together_dataframes are required.")

    scenario_alone = f"{device_name} Alone"
    scenario_together = f"{device_name} Together"

    data_list = []

    # Gather data from both scenarios at the given fps
    for points in experiment_number_of_points:
        key = f"{fps}_{points}"

        # Alone scenario
        if key in alone_dataframes:
            df_alone = alone_dataframes[key].copy()
            df_alone["scenario"] = scenario_alone
            df_alone["points"] = points
            data_list.append(df_alone)

        # Together scenario
        if key in together_dataframes:
            df_together = together_dataframes[key].copy()
            df_together["scenario"] = scenario_together
            df_together["points"] = points
            data_list.append(df_together)

    # Concatenate all the data for plotting
    if not data_list:
        print("No data found for the given parameters.")
        return

    combined_df = pd.concat(data_list, ignore_index=True)

    # Check if required columns exist
    if "fps_current" not in combined_df.columns or "fps_moving_media" not in combined_df.columns:
        raise ValueError("The required columns 'fps_current' and 'fps_moving_media' do not exist in the DataFrames.")

    sns.set_style("whitegrid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # First subplot: fps_current
    sns.lineplot(
        data=combined_df,
        x="Frame Number",
        y="fps_current",
        hue="points",
        style="scenario",
        ax=ax1
    )
    ax1.set_title(f"fps_current at {fps} FPS for {device_name}")
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("fps_current (frames/sec)")
    if y_min is not None and y_max is not None:
        ax1.set_ylim(y_min, y_max)
    ax1.legend(title="Points / Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Second subplot: fps_moving_media
    sns.lineplot(
        data=combined_df,
        x="Frame Number",
        y="fps_moving_media",
        hue="points",
        style="scenario",
        ax=ax2,
        legend=False
    )
    ax2.set_title(f"fps_moving_media at {fps} FPS for {device_name}")
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("fps_moving_media (frames/sec)")
    if y_min is not None and y_max is not None:
        ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
    
    
def load_mpd_experiments(root_dir='./experiments/raw'):
    """
    Recursively loads all CSV files (only those whose names start with 'All-Data')
    within root_dir. Returns a single concatenated DataFrame with:
      - 'Network': inferred ('5G' or 'Wifi') from the CSV filename
      - 'Concurrent': boolean, True if 'Concurrent' is in the filename
      - Columns stripped of leading/trailing spaces
      - Columns reordered so it will be: ['Test Name', 'Device Name', 'Concurrent', 'Network', ...]
    """
    dfs = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Only process files matching "All-Data*.csv"
            if file.startswith("All-Data") and file.endswith(".csv"):
                print('Processing file', file)
                
                testbed = root.split('/')[3]
                transport = root.split('/')[4]
                
                full_path = os.path.join(root, file)
                
                # Parse the file name (without .csv)
                filename_no_ext = os.path.splitext(file)[0]
                
                # Determine the Network from the file name
                # e.g., "All-Data-5G..." => Network = '5G'
                #       "All-Data-Wifi..." => Network = 'Wifi'
                if "5G" in filename_no_ext:
                    network = "5G"
                elif "Wifi" in filename_no_ext:
                    network = "Wifi"
                else:
                    # If neither 5G nor Wifi found, skip or set a default
                    continue  # or network = "Unknown"
                
                # Determine the Concurrent boolean
                # True if the word "Concurrent" is anywhere in the filename
                concurrent = "Concurrent" in filename_no_ext
                
                # Read the CSV into a DataFrame
                df = pd.read_csv(full_path, dtype=str, na_values=['nan', 'undefined'])
                
                # Strip extra spaces from column names
                # e.g. ' Frame FPS' becomes 'Frame FPS'
                df.rename(columns=lambda x: x.strip(), inplace=True)
                
                # Unify buffer column's names, e.g., Bf1 In = bf1_in_timestamp
                if "bf1 in" in df.columns.str.lower():
                    df.rename(columns={"Bf1 In": "bf1_in_timestamp"}, inplace=True)
                if "bf1 out" in df.columns.str.lower():
                    df.rename(columns={"Bf1 out": "bf1_out_timestamp"}, inplace=True)
                if "bf2 in" in df.columns.str.lower():
                    df.rename(columns={"Bf2 In": "bf2_in_timestamp"}, inplace=True)
                if "bf2 out" in df.columns.str.lower():
                    df.rename(columns={"Bf2 out": "bf2_out_timestamp"}, inplace=True)
                
                # Add our two new columns
                df["Network"] = network
                df["Concurrent"] = concurrent
                df["Testbed"] = testbed
                df["Transport"] = transport
                # Reorder columns: "Test Name", "Device Name", "Concurrent", "Network", ...
                # 1. Define the desired front columns in order
                front_cols = ["Test Name", "Device Name", "Concurrent", "Network"]
                
                # 2. Build a new column order, ensuring we only reorder if they exist
                #    and preserving the rest of the columns in their original order.
                existing_cols = list(df.columns)
                
                # We'll collect the front columns that actually exist (some CSVs may not have both "Test Name" or "Device Name")
                front_cols_in_df = [col for col in front_cols if col in existing_cols]
                
                # The remaining columns are all those not in front_cols_in_df, in the original order
                remaining_cols = [col for col in existing_cols if col not in front_cols_in_df]
                
                # Final column order
                new_order = front_cols_in_df + remaining_cols
                df = df[new_order]
                dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df