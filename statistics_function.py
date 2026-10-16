import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import warnings
import ast
import re
import pickle
import glob


def unpack_uncertainty_columns(df):
    """
    Scans a DataFrame for columns containing '±'. 
    Replaces the original column with the float mean, and dynamically 
    inserts a 'std_columnName' immediately to its right.
    """
    if df is None or df.empty:
        return df

    # Create a copy to prevent SettingWithCopy warnings
    df_clean = df.copy()
    
    # We must iterate over a static list of columns, because we are 
    # going to dynamically insert new columns into the DataFrame during the loop
    original_cols = list(df_clean.columns)
    
    for col in original_cols:
        # 1. Verify if the column is an object/string type
        if df_clean[col].dtype == 'O':
            
            # 2. Check if the '±' symbol exists anywhere in this specific column
            valid_mask = df_clean[col].notna()
            if valid_mask.any() and df_clean.loc[valid_mask, col].astype(str).str.contains('±').any():
                
                # 3. Locate the exact index of this column to insert the new one right next to it
                col_idx = df_clean.columns.get_loc(col)
                
                # 4. Vectorized Split: Splits '10.48 ± 0.32' into two separate columns
                split_df = df_clean[col].astype(str).str.split('±', expand=True)
                
                # 5. Convert to rigorous floats, forcing 'nan' strings into actual np.nan
                mean_vals = pd.to_numeric(split_df[0].str.strip(), errors='coerce')
                
                if split_df.shape[1] > 1:
                    std_vals = pd.to_numeric(split_df[1].str.strip(), errors='coerce')
                else:
                    # Failsafe: If a row lacked a '±', its standard deviation is mathematically undefined
                    std_vals = np.nan 
                
                # 6. Overwrite the original column with the means, and insert the standard deviations
                df_clean[col] = mean_vals
                df_clean.insert(loc=col_idx + 1, column=f"std_{col}", value=std_vals)
                
    return df_clean





def load_local_silso_data(file_path):
    """
    Loads the official Monthly Mean Total Sunspot Number 
    from a locally downloaded SILSO CSV file.
    """
    print(f"Loading empirical Solar Cycle data from local file: {file_path}...")
    
    # SILSO V2.0 CSV formatting specifications (no header in the raw file)
    cols = ['Year', 'Month', 'Decimal_Year', 'SSN', 'StdDev', 'Obs_Count', 'Marker']
    
    if not os.path.exists(file_path):
        print(f"Warning: The file '{file_path}' was not found. Background will be blank.")
        return pd.DataFrame()
        
    try:
        # The SILSO CSV uses semicolons as delimiters
        df_ssn = pd.read_csv(file_path, sep=';', names=cols)
        
        # Filter out missing data (SILSO marks missing months with -1)
        df_ssn = df_ssn[df_ssn['SSN'] >= 0]
        return df_ssn
    except Exception as e:
        print(f"CRITICAL: Failed to parse the local SILSO CSV. ({e})")
        return pd.DataFrame()





def event_dictionary(folder_path):
    
    print(f"Scanning directory: {folder_path}")
    mother_dict = {}
    filecount = 0
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pkl"):
            continue
            
        file_path = os.path.join(folder_path, filename)
        filecount = filecount + 1
        with open(file_path, "rb") as file:
            data_payload = pickle.load(file)
            
        event_id = data_payload.get('event_id')
        
        if not event_id:
            print(f"Warning: No 'event_id' found in {filename}. Skipping.")
            continue
        
        # Initialize the event-level dictionary structure if it doesn't exist
        if event_id not in mother_dict:
            mother_dict[event_id] = {
                'ME': None,
                'Sheath': None,
                'Metadata': {} 
            }
            
        # =========================================================
        # TENSOR ROUTING & METADATA PRESERVATION
        # =========================================================
        if 'df_me' in data_payload:
            clean_me_df = unpack_uncertainty_columns(data_payload['df_me'])
            mother_dict[event_id]['ME'] = clean_me_df
            mother_dict[event_id]['Metadata']['spacecraft'] = data_payload.get('spacecraft')
            mother_dict[event_id]['Metadata']['r_AU'] = data_payload.get('r')
            
        elif 'df_sheath' in data_payload:
            clean_sheath_df = unpack_uncertainty_columns(data_payload['df_sheath'])
            mother_dict[event_id]['Sheath'] = clean_sheath_df
            mother_dict[event_id]['Metadata']['spacecraft'] = data_payload.get('spacecraft')
            mother_dict[event_id]['Metadata']['r_AU'] = data_payload.get('r')
            
        print(f"Loaded {filecount}th file: {filename}")
            
    print(f"Successfully mapped {len(mother_dict)} distinct ICME events into the master dictionary.")
    return mother_dict


# mother_dict = {"event_id" : {'ME': df_me,
#                             'Sheath' : df_sheath,
#                             'Metadata' : {
#                                          'spacecraft' : spacecraft_name,
#                                           'r_AU' : r
#                                          }
#                              }
#               }




def ingest_hathaway_rgo_data(path):
    """
    Ingests the NASA/RGO space-delimited text files. 
    Accepts either a single .txt file path or a folder path containing multiple 'g*.txt' files.
    """
    master_records = []
    
    # Determine if the input is a single file or a directory
    if os.path.isfile(path):
        file_list = [path]
    elif os.path.isdir(path):
        file_list = glob.glob(os.path.join(path, "g*.txt"))
    else:
        print(f"CRITICAL: The path '{path}' is invalid.")
        return pd.DataFrame()
        
    print(f"Scanning {len(file_list)} RGO text file(s)...")
    
    for file in file_list:
        try:
            # Extract specific columns based on the RGO Daily format
            df = pd.read_csv(file, sep=r'\s+', header=None, comment='#', 
                             usecols=[0, 1, 2, 8, 12], 
                             names=['Year', 'Month', 'Day', 'Area', 'Latitude'])
            
            master_records.append(df)
        except Exception as e:
            print(f"Warning: Failed to parse {file}. Error: {e}")
            
    if not master_records:
        return pd.DataFrame()
        
    df_master = pd.concat(master_records, ignore_index=True)
    
    # 1. CLEANING THE TENSOR
    df_master = df_master.dropna()
    df_master = df_master.apply(pd.to_numeric, errors='coerce')
    
    # Filter out spotless days (Area = 0)
    df_master = df_master[df_master['Area'] > 0].copy()
    
    # 2. TEMPORAL CONVERSION
    # Convert Year, Month, Day into a mathematically continuous Decimal Year
    # We use a highly efficient vectorized approximation suitable for 150-year spans
    df_master['Decimal_Year'] = df_master['Year'] + (df_master['Month'] - 1) / 12 + (df_master['Day'] - 1) / 365.25
    
    # 3. INSTRUMENTAL CALIBRATION
    # Apply Hathaway's 1.4 multiplier to all USAF/NOAA data recorded after 1976
    mask_modern = df_master['Year'] >= 1977
    df_master.loc[mask_modern, 'Area'] = df_master.loc[mask_modern, 'Area'] * 1.4
    
    print(f"Success: Extracted {len(df_master)} valid sunspot observations.")
    return df_master


# df_rgo = ingest_hathaway_rgo_data(butterfly_path)





ATTRIBUTE_LABELS = {
    "B": "Magnetic Field Intensity (B) (nT)",
    "Np": "Proton Density (Np) ($cm^{-3}$)",
    "Vsw": "Bulk velocity (Vsw) (km/s)",
    "Tp": "Proton Temperature (Tp) (K)",
    "Va": "Alfvén Speed (Va) (km/s)",
    "M_A": "Alfvén Mach Number ($M_A$)",
    "beta": "Plasma Beta (β)",
    "P_dyn": "Dynamic Pressure (P_dyn) (nPa)",
    "P_T": "Total Pressure (P_T) (nPa)",
    "T_r": "Temperature ratio ($T_r$)",
    "S": "Entropy (S)",
    "Na/Np": "Alpha to proton ratio (Na/Np)",
    "rmsB_over_B": "Normalized magnetic field fluctuation amplitude",
    "deltaB": "Fluctuation in B"
}





# Suppress minor seaborn/matplotlib warnings for clean execution
warnings.filterwarnings("ignore", category=FutureWarning)

# Set global visual style for academic plotting
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

def attr_vs_probability(mother_dict, event_type, radial_bin, attribute_key):
    """
    Generates Probability Frequency Polygons for attributes.
    Features: Ultra-refined binning, bold lines, translucent area fills, and strict inward ticks.
    """
        
    # 1. INPUT VALIDATION 
    try:
        axis_label = ATTRIBUTE_LABELS[attribute_key]
    except NameError:
        axis_label = attribute_key 
        
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
        
    title_event_label = "ICME" if ((event_type.lower() == "both") | (event_type.upper() == "ICME")) else event_type.upper()
    r_min, r_max = radial_bin
    
    print(f"\n--- Aggregating '{attribute_key}' for {title_event_label} in {r_min} - {r_max} AU ---")
    
    # =========================================================
    # 2. TENSOR AGGREGATION
    # =========================================================
    aggregated_me = []
    aggregated_sheath = []
    count_me = 0
    count_sheath = 0
    for event_id, event_data in mother_dict.items():
        r_val = event_data.get('Metadata', {}).get('r_AU', None)
        
        if r_val is None or not (r_min <= r_val <= r_max):
            continue
            
        if event_type.lower() in ['me', 'both', 'icme'] and event_data.get('ME') is not None:
            aggregated_me.append(event_data['ME'])
            count_me += 1
        if event_type.lower() in ['sheath', 'both', 'icme'] and event_data.get('Sheath') is not None:
            aggregated_sheath.append(event_data['Sheath'])
            count_sheath += 1

    if not aggregated_me and not aggregated_sheath:
        print(f"CRITICAL: No valid data found in the {r_min} - {r_max} AU bin.")
        return

    # Master DataFrames
    df_me = pd.concat(aggregated_me, ignore_index=True) if aggregated_me else pd.DataFrame()
    df_sh = pd.concat(aggregated_sheath, ignore_index=True) if aggregated_sheath else pd.DataFrame()
    
    # 3. EXTRACT SPECIFIC ARRAY
    array_me = pd.to_numeric(df_me[attribute_key], errors='coerce').dropna() if attribute_key in df_me.columns else pd.Series(dtype=float)
    array_sh = pd.to_numeric(df_sh[attribute_key], errors='coerce').dropna() if attribute_key in df_sh.columns else pd.Series(dtype=float)

    if array_me.empty and array_sh.empty:
        print(f"CRITICAL: The column '{attribute_key}' contains no valid numeric data in this radial bin.")
        return

    # =========================================================
    # 4. VISUALIZATION ARCHITECTURE (Hybrid Aesthetic)
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    combined_data = pd.concat([array_me, array_sh])
    min_val, max_val = combined_data.min(), combined_data.max()
    
    # REFINED BINNING: Using 120 bins to simulate a continuous, smooth line
    bins = np.linspace(min_val, max_val, 120)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # --- Plot Sheath (Black Curve with Gray Fill) ---
    if not array_sh.empty and event_type.lower() in ['sheath', 'both', 'icme']:
        counts_sh, _ = np.histogram(array_sh, bins=bins)
        prob_sh = counts_sh / len(array_sh) 
        
        # Bold Line
        ax.plot(bin_centers, prob_sh, color="black", linewidth=2.5, label=f"Sheath ({count_sheath})", zorder=3)
        # Translucent Area Fill
        ax.fill_between(bin_centers, prob_sh, 0, color="gray", alpha=0.3, zorder=2)

    # --- Plot ME/ICME (Red Curve with Light Red Fill) ---
    if not array_me.empty and event_type.lower() in ['me', 'both', 'icme']:
        counts_me, _ = np.histogram(array_me, bins=bins)
        prob_me = counts_me / len(array_me) 
        
        # Bold Line
        ax.plot(bin_centers, prob_me, color="#CC3333", linewidth=2.5, label=f"ME ({count_me})", zorder=4)
        # Translucent Area Fill
        ax.fill_between(bin_centers, prob_me, 0, color="#CC3333", alpha=0.3, zorder=1)

    # =========================================================
    # 5. FORMATTING & AESTHETICS 
    # =========================================================
    ax.set_xlabel(f"{axis_label}", fontweight='bold', fontsize=13)
    ax.set_ylabel("Probability", fontweight='bold', fontsize=13)
    
    # Lock the X-axis tightly to the data minimum and maximum
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(bottom=0)
    
    # Advanced Tick Formatting (Inward pointing, top and right enabled)
    ax.tick_params(axis='both', which='major', direction='in', length=7, width=1.2, top=True, right=True, labelsize=11)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0, top=True, right=True)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # Ensure the plot has a hard black border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Grid and Title
    title_str = f"Probability Distribution: {axis_label}\n{title_event_label} ({r_min} - {r_max} AU)"
    ax.set_title(title_str, fontweight='bold', fontsize=14, pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, edgecolor='black', fontsize=12)
    
    plt.tight_layout()
    
    # plt.show()
    
    return fig
    
    
    # =========================================================
    # 6. EXPORT
    # =========================================================
    safe_filename = attribute_key.replace("/", "_")
    # save_path = os.path.join(output_dir, f"Dist_Filled_{safe_filename}_{title_event_label}_{r_min}_{r_max}AU.png")
    # plt.savefig(save_path, facecolor='white', bbox_inches='tight')
    
    # if str(show_plot).lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)

    # print(f"Success: plot saved to '{save_path}'.")


# event_type   : 'ME', 'Sheath', or 'Both'/'ICME'
# radial_bin   : Tuple of floats (e.g., (0.0, 0.5), (0.8, 1.2), (4.0, 5.5))
# attribute_key: 
#   'B'           (Magnetic Field Intensity)
#   'Np'          (Proton Density)
#   'Vsw'         (Bulk velocity)
#   'Tp'          (Proton Temperature)
#   'Va'          (Alfvén Speed)
#   'M_A'         (Alfvén Mach Number)
#   'beta'        (Plasma Beta)
#   'P_dyn'       (Dynamic Pressure)
#   'P_T'         (Total Pressure)
#   'T_r'         (Temperature ratio)
#   'S'           (Entropy)
#   'Na/Np'       (Alpha to proton ratio)
#   'rmsB_over_B' (Normalized magnetic field fluctuation amplitude)
#   'deltaB'      (Fluctuation in B)
# show_plot : 'y' or anything else
# =========================================================

# Execution Example:
# plot_single_radial_pdf(mother_dict, event_type="Both", radial_bin=(0.8, 1.2), attribute_key="beta", show_plot = 'y')





def t_theta_attr_R_2d(mother_dict, attribute_key, event_type, silso_csv):
    """
    Generates a 2D visualization of ICME events using discrete radial binning.
    X: Time | Y: Latitude | Size: Categorical Radial Bin (Large R = Large Disc).
    Color: Attribute Value (Light = Low, Dark = High).
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    target_keys = []
    if event_type.lower() in ['me', 'icme', 'both']: target_keys.append('ME')
    if event_type.lower() in ['sheath', 'icme', 'both']: target_keys.append('Sheath')
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()

    print(f"\n--- Constructing Stratified Perspective for {attribute_key} ({title_event_label}) ---")

    # =========================================================
    # 1. EVENT-LEVEL DATA EXTRACTION (Corrected Timeline)
    # =========================================================
    records = []
    
    for event_id, event_data in mother_dict.items():
        # 1. Extract base Year from the dictionary key
        match_year = re.search(r'(19|20)\d{2}', event_id)
        if not match_year:
            continue
        year = float(match_year.group(0))

        # 2. FETCH R DIRECTLY FROM METADATA (Bypassing Location Tuple)
        metadata = event_data.get('Metadata', {})
        event_R = metadata.get('r_AU') 
        
        if event_R is None:
            continue

        event_attr_pool, event_theta_pool, event_doy_pool = [], [], []

        for key in target_keys:
            df = event_data.get(key)
            if df is None or attribute_key not in df.columns or 'Location' not in df.columns:
                continue

            # 3. Extract precise DOY from Epoch_ID
            if 'Epoch_ID' in df.columns:
                for epoch_val in df['Epoch_ID'].dropna():
                    match_doy = re.search(r'_(\d+\.\d+)$', str(epoch_val))
                    if match_doy:
                        event_doy_pool.append(float(match_doy.group(1)))

            # Extract Attributes
            raw_vals = pd.to_numeric(df[attribute_key], errors='coerce').dropna().tolist()
            event_attr_pool.extend(raw_vals)

            # Extract Spatial Vectors (Index 2 for Theta only)
            for loc in df['Location'].dropna():
                loc_str = str(loc).replace('nan', 'None')
                try:
                    t = ast.literal_eval(loc_str)
                    if t[2] is not None:
                        event_theta_pool.append(float(t[2]))
                except:
                    pass
        
        # 4. Compile mathematical mean for the event
        if event_attr_pool and event_theta_pool:
            if event_doy_pool:
                mean_doy = np.mean(event_doy_pool)
                dec_year = year + (mean_doy / 365.25)
            else:
                continue # Skip if no precise temporal data exists

            records.append({
                'Decimal_Year': dec_year,
                'R': float(event_R),          # Using exact Metadata R
                'Theta': np.mean(event_theta_pool),
                'Attr_Mean': np.mean(event_attr_pool)
            })

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        print("CRITICAL: No valid spatial/attribute data found for this combination.")
        return

    # =========================================================
    # 2. DISCRETE CATEGORICAL BINNING
    # =========================================================
    # Define the strict mathematical boundaries for R
    bins = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf]
    
    # Define the label definitions for the legend
    labels = ['< 0.5 AU', '0.5 - 1.0 AU', '1.0 - 2.0 AU', '2.0 - 3.0 AU', '3.0 - 4.0 AU', '4.0 - 5.0 AU', '5.0+ AU']
    
    # Define the corresponding marker sizes (Exponentially increasing to guarantee visual distinction)
    sizes = [30, 80, 180, 320, 500, 750, 1000]
    
    # pd.cut categorizes the data into bins, and map assigns the specific size integer
    df_plot['R_Category'] = pd.cut(df_plot['R'], bins=bins, labels=labels, right=True)
    size_mapping = dict(zip(labels, sizes))
    df_plot['Disc_Size'] = df_plot['R_Category'].map(size_mapping).astype(float)

    # CRITICAL Z-ORDERING: Plot the largest sizes first so they sit in the background
    df_plot = df_plot.sort_values(by='Disc_Size', ascending=False)

    # =========================================================
    # 3. VISUALIZATION ARCHITECTURE
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)

    min_yr = np.floor(df_plot['Decimal_Year'].min())
    max_yr = np.ceil(df_plot['Decimal_Year'].max())

    # --- Background: SILSO Solar Cycle (Standard Gray) ---
    try:
        df_ssn = load_local_silso_data(silso_csv)
        if not df_ssn.empty:
            mask = (df_ssn['Decimal_Year'] >= min_yr - 1) & (df_ssn['Decimal_Year'] <= max_yr + 1)
            silso_clipped = df_ssn[mask]
            
            ax_bg = ax1.twinx()
            ax_bg.fill_between(silso_clipped['Decimal_Year'], 0, silso_clipped['SSN'], 
                               color='gray', alpha=0.15, zorder=0)
            ax_bg.set_ylim(0, silso_clipped['SSN'].max() * 2.5)
            ax_bg.axis('off')
    except Exception as e:
        print(f"Warning: Could not load SILSO background ({e})")

    # --- Solar Equator (The Sun) ---
    ax1.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.6, zorder=1, label='Solar Equator')

    # --- Foreground: The Stratified Discs ---
    # Colormap 'YlOrRd' anchors low values at light yellow and scales up to dark red
    scatter = ax1.scatter(
        df_plot['Decimal_Year'], 
        df_plot['Theta'], 
        c=df_plot['Attr_Mean'], 
        s=df_plot['Disc_Size'], 
        cmap='viridis_r',       
        alpha=0.75,          
        edgecolors='black', 
        linewidth=0.5,
        zorder=3
    )

    # =========================================================
    # 4. FORMATTING & AESTHETICS
    # =========================================================
    ax1.set_title(f"Solar Cycle, Latitude, R, and {attribute_key} ({title_event_label})", 
                  fontweight='bold', fontsize=16, pad=15)
    
    ax1.set_xlabel("Time [Years]", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Latitude (θ) [Degrees]", fontweight='bold', fontsize=12)
    
    max_lat = max(abs(df_plot['Theta'])) + 10
    ax1.set_ylim(-max_lat, max_lat)
    
    ax1.set_xlim(min_yr, max_yr)
    ax1.set_xticks(np.arange(min_yr, max_yr + 1, 2)) 
    ax1.tick_params(axis='x', rotation=45)
    
    ax1.grid(True, linestyle=':', color='gray', alpha=0.4, zorder=0)

    # --- Colorbar ---
    cbar = fig.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label(f'Mean {attribute_key} Value', fontweight='bold', fontsize=11)
    
    # --- Custom Categorical Legend ---
    # --- Custom Categorical Legend ---
    from matplotlib.lines import Line2D
    
    # 1. Create a custom handle for the Solar Equator
    custom_handles = [
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Solar Equator')
    ]
    
    # 2. Iterate through the active classes to build proxy circles
    active_labels = df_plot['R_Category'].dropna().unique()
    for label, size in zip(labels, sizes):
        if label in active_labels:
            # We use a dummy scatter plot to generate the exact size for the legend
            proxy_circle = ax1.scatter([], [], s=size, c='lightgray', alpha=0.8, edgecolors='black', label=label)
            custom_handles.append(proxy_circle)
            
    # 3. Anchor the legend to the UPPER LEFT to completely avoid the colorbar
    ax1.legend(
        handles=custom_handles, 
        # title="Radial Domain (R)", 
        loc='upper left', 
        frameon=True, 
        facecolor='white', 
        edgecolor='black', 
        labelspacing=1.2
    )

    plt.tight_layout()
    
    # =========================================================
    # 5. EXPORT
    # =========================================================
    safe_filename = attribute_key.replace("/", "_")
    # save_path = os.path.join(output_dir, f"Perspective_Stratified_{safe_filename}_{title_event_label}.png")
    # plt.savefig(save_path, facecolor='white', edgecolor='none')
    # print(f"Plot successfully saved to: {save_path}")
    
    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)
    
    
    # plt.show()
    return fig
    




def t_theta_attr_R_3d(mother_dict, attribute_key, event_type, silso_csv):
    """
    Generates a true 3D spatial visualization of ICME events.
    X: Time | Y: Latitude | Z: Radial Distance.
    Size: Fixed | Color: Attribute Value.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    target_keys = []
    if event_type.lower() in ['me', 'icme', 'both']: target_keys.append('ME')
    if event_type.lower() in ['sheath', 'icme', 'both']: target_keys.append('Sheath')
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()

    print(f"\n--- Constructing 3D Spatial Topology for {attribute_key} ({title_event_label}) ---")

    # =========================================================
    # 1. EVENT-LEVEL DATA EXTRACTION
    # =========================================================
    records = []
    
    for event_id, event_data in mother_dict.items():
        # 1. Extract base Year from the dictionary key
        match_year = re.search(r'(19|20)\d{2}', event_id)
        if not match_year:
            continue
        year = float(match_year.group(0))

        # FETCH R DIRECTLY FROM METADATA (3D-Specific)
        metadata = event_data.get('Metadata', {})
        event_R = metadata.get('r_AU') 
        
        if event_R is None:
            continue

        event_attr_pool, event_theta_pool, event_doy_pool = [], [], []

        for key in target_keys:
            df = event_data.get(key)
            if df is None or attribute_key not in df.columns or 'Location' not in df.columns:
                continue

            # 2. Extract precise DOY from Epoch_ID
            if 'Epoch_ID' in df.columns:
                for epoch_val in df['Epoch_ID'].dropna():
                    match_doy = re.search(r'_(\d+\.\d+)$', str(epoch_val))
                    if match_doy:
                        event_doy_pool.append(float(match_doy.group(1)))

            # Extract Attributes
            raw_vals = pd.to_numeric(df[attribute_key], errors='coerce').dropna().tolist()
            event_attr_pool.extend(raw_vals)

            # Extract Spatial Vectors (Index 2 for Theta)
            for loc in df['Location'].dropna():
                loc_str = str(loc).replace('nan', 'None')
                try:
                    t = ast.literal_eval(loc_str)
                    if t[2] is not None:
                        event_theta_pool.append(float(t[2]))
                except:
                    pass
        
        # 3. Compile mathematical mean for the event
        if event_attr_pool and event_theta_pool:
            
            if event_doy_pool:
                mean_doy = np.mean(event_doy_pool)
                dec_year = year + (mean_doy / 365.25)
            else:
                continue # Skip if no precise temporal data exists

            records.append({
                'Decimal_Year': dec_year,
                'R': float(event_R),
                'Theta': np.mean(event_theta_pool),
                'Attr_Mean': np.mean(event_attr_pool)
            })

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        print("CRITICAL: No valid spatial/attribute data found for this combination.")
        return

    # =========================================================
    # 2. 3D VISUALIZATION ARCHITECTURE
    # =========================================================
    fig = plt.figure(figsize=(16, 10), dpi=150)
    ax1 = fig.add_subplot(111, projection='3d') 

    min_yr = np.floor(df_plot['Decimal_Year'].min())
    max_yr = np.ceil(df_plot['Decimal_Year'].max())
    max_lat = max(abs(df_plot['Theta'])) + 10
    max_R_plot = df_plot['R'].max()

    # --- Background: SILSO Solar Cycle (Projected on Z=0 Plane) ---
    try:
        df_ssn = load_local_silso_data(silso_csv)
        if not df_ssn.empty:
            mask = (df_ssn['Decimal_Year'] >= min_yr - 1) & (df_ssn['Decimal_Year'] <= max_yr + 1)
            silso_clipped = df_ssn[mask]
            
            # Scale SSN amplitude to fit within Latitude bounds
            max_ssn = silso_clipped['SSN'].max()
            scaled_ssn = -max_lat + (silso_clipped['SSN'] / max_ssn) * (max_lat * 2)
            
            # zs=0 locks the solar cycle drawing to the floor (Theta-Year plane)
            ax1.plot(silso_clipped['Decimal_Year'], scaled_ssn, zs=0, zdir='z', 
                     color='gray', alpha=0.3, linewidth=2, label='Scaled Solar Cycle')
    except Exception as e:
        print(f"Warning: Could not load SILSO background ({e})")

    # --- Solar Equator (The Sun) ---
    ax1.plot([min_yr, max_yr], [0, 0], zs=0, zdir='z', color='gold', linewidth=3, alpha=0.8, label='Solar Equator (Sun)')

    # --- Foreground: The 3D Scatter (Fixed Size) ---
    scatter = ax1.scatter(
        df_plot['Decimal_Year'], 
        df_plot['Theta'], 
        df_plot['R'],           
        c=df_plot['Attr_Mean'], 
        s=40,                   # Fixed size for all events
        cmap='viridis_r',          
        alpha=0.85,          
        edgecolors='black', 
        linewidth=0.5,
        depthshade=True         
    )

    # =========================================================
    # 3. FORMATTING & AESTHETICS
    # =========================================================
    ax1.set_title(f"Solar Cycle, Latitude, R, and {attribute_key} ({title_event_label})", 
                  fontweight='bold', fontsize=16, pad=20)
    
    ax1.set_xlabel("Time [Years]", fontweight='bold', fontsize=11, labelpad=10)
    ax1.set_ylabel("Latitude (θ) [Degrees]", fontweight='bold', fontsize=11, labelpad=10)
    ax1.set_zlabel("Radial Distance (R) [AU]", fontweight='bold', fontsize=11, labelpad=10)
    
    ax1.set_ylim(-max_lat, max_lat)
    ax1.set_zlim(0, max_R_plot * 1.05) 
    
    ax1.set_xlim(min_yr, max_yr)
    ax1.set_xticks(np.arange(min_yr, max_yr + 1, 3)) 
    ax1.tick_params(axis='x', rotation=45)

    ax1.view_init(elev=20, azim=-60)

    # --- Colorbar ---
    cbar = fig.colorbar(scatter, ax=ax1, pad=0.05, shrink=0.7)
    cbar.set_label(f'Mean {attribute_key} Value', fontweight='bold', fontsize=11)
    
    # --- Custom Legend ---
    custom_handles = [
        Line2D([0], [0], color='gold', linewidth=3, label='Solar Equator'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.4, label='Solar Cycle Timeline')
    ]
    
    ax1.legend(handles=custom_handles, loc='upper left', 
               bbox_to_anchor=(0.0, 1.0), frameon=True, facecolor='white', edgecolor='black', labelspacing=1.2)

    plt.tight_layout()
    
    # =========================================================
    # 4. EXPORT
    # =========================================================
    safe_filename = attribute_key.replace("/", "_")
    # save_path = os.path.join(output_dir, f"Perspective_3D_{safe_filename}_{title_event_label}.png")
    # plt.savefig(save_path, facecolor='white', edgecolor='none')
    # print(f"Plot successfully saved to: {save_path}")
    
    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)
    
    # plt.show()
    return fig





def t_phi_attr_R_2d(mother_dict, attribute_key, event_type, silso_csv):
    """
    Generates a 2D visualization of ICME events using discrete radial binning.
    Correctly extracts precise DOY from the Epoch_ID column.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    target_keys = ['ME', 'Sheath'] if event_type.lower() in ['both', 'icme'] else [event_type.upper()]
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()

    print(f"\n--- Constructing Stratified Perspective (Longitude) for {attribute_key} ({title_event_label}) ---")

    # =========================================================
    # 1. EVENT-LEVEL DATA EXTRACTION
    # =========================================================
    records = []
    
    for event_id, event_data in mother_dict.items():
        # 1. Extract base Year from the dictionary key
        match_year = re.search(r'(19|20)\d{2}', event_id)
        if not match_year:
            continue
        year = float(match_year.group(0))

        # 2. FETCH R DIRECTLY FROM METADATA (Bypassing Location Tuple)
        metadata = event_data.get('Metadata', {})
        event_R = metadata.get('r_AU') 
        
        if event_R is None:
            continue

        event_attr_pool, event_phi_pool, event_doy_pool = [], [], []

        for key in target_keys:
            df = event_data.get(key)
            if df is None or attribute_key not in df.columns or 'Location' not in df.columns:
                continue

            # 3. Extract precise DOY from Epoch_ID
            if 'Epoch_ID' in df.columns:
                for epoch_val in df['Epoch_ID'].dropna():
                    match_doy = re.search(r'_(\d+\.\d+)$', str(epoch_val))
                    if match_doy:
                        event_doy_pool.append(float(match_doy.group(1)))

            # Extract Attributes
            raw_vals = pd.to_numeric(df[attribute_key], errors='coerce').dropna().tolist()
            event_attr_pool.extend(raw_vals)

            # Extract Spatial Vectors (Index 2 for Theta only)
            for loc in df['Location'].dropna():
                loc_str = str(loc).replace('nan', 'None')
                try:
                    t = ast.literal_eval(loc_str)
                    if t[1] is not None:
                        event_phi_pool.append(float(t[1]))
                except:
                    pass
        
        # 4. Compile mathematical mean for the event
        if event_attr_pool and event_phi_pool:
            if event_doy_pool:
                mean_doy = np.mean(event_doy_pool)
                dec_year = year + (mean_doy / 365.25)
            else:
                continue # Skip if no precise temporal data exists

            records.append({
                'Decimal_Year': dec_year,
                'R': float(event_R),          # Using exact Metadata R
                'Phi': np.mean(event_phi_pool),
                'Attr_Mean': np.mean(event_attr_pool)
            })
    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        print("CRITICAL: No valid spatial/attribute data found.")
        return

    # =========================================================
    # 2. DISCRETE CATEGORICAL BINNING
    # =========================================================
    bins = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf]
    labels = ['< 0.5 AU', '0.5 - 1.0 AU', '1.0 - 2.0 AU', '2.0 - 3.0 AU', '3.0 - 4.0 AU', '4.0 - 5.0 AU', '5.0+ AU']
    sizes = [30, 80, 180, 320, 500, 750, 1000]
    
    df_plot['R_Category'] = pd.cut(df_plot['R'], bins=bins, labels=labels, right=True)
    df_plot['Disc_Size'] = df_plot['R_Category'].map(dict(zip(labels, sizes))).astype(float)
    df_plot = df_plot.sort_values(by='Disc_Size', ascending=False)

    # =========================================================
    # 3. VISUALIZATION ARCHITECTURE
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)
    min_yr, max_yr = np.floor(df_plot['Decimal_Year'].min()), np.ceil(df_plot['Decimal_Year'].max())

    try:
        df_ssn = load_local_silso_data(silso_csv)
        if not df_ssn.empty:
            mask = (df_ssn['Decimal_Year'] >= min_yr - 1) & (df_ssn['Decimal_Year'] <= max_yr + 1)
            silso_clipped = df_ssn[mask]
            ax_bg = ax1.twinx()
            ax_bg.fill_between(silso_clipped['Decimal_Year'], 0, silso_clipped['SSN'], color='gray', alpha=0.15, zorder=0)
            ax_bg.set_ylim(0, silso_clipped['SSN'].max() * 2.5)
            ax_bg.axis('off')
    except Exception as e: pass

    ax1.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.6, zorder=1, label='Prime Meridian')

    scatter = ax1.scatter(df_plot['Decimal_Year'], df_plot['Phi'], c=df_plot['Attr_Mean'], 
                          s=df_plot['Disc_Size'], cmap='viridis_r', alpha=0.75, edgecolors='black', linewidth=0.5, zorder=3)

    # =========================================================
    # 4. FORMATTING & AESTHETICS 
    # =========================================================
    ax1.set_title(f"Solar Cycle, Longitude, R, and {attribute_key} ({title_event_label})", fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel("Time [Years]", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Longitude (φ) [Degrees]", fontweight='bold', fontsize=12)
    
    # DYNAMIC Y-AXIS FRAMING
    y_min_data = df_plot['Phi'].min()
    y_max_data = df_plot['Phi'].max()
    ax1.set_ylim(y_min_data - 20, y_max_data + 20)
    
    ax1.set_xlim(min_yr, max_yr)
    ax1.set_xticks(np.arange(min_yr, max_yr + 1, 2)) 
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle=':', color='gray', alpha=0.4, zorder=0)

    # --- Legends & Colorbars ---
    cbar = fig.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label(f'Mean {attribute_key} Value', fontweight='bold', fontsize=11)
    
    custom_handles = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Zero Line')]
    for label, size in zip(labels, sizes):
        if label in df_plot['R_Category'].dropna().unique():
            custom_handles.append(ax1.scatter([], [], s=size, c='lightgray', alpha=0.8, edgecolors='black', label=label))
            
    ax1.legend(handles=custom_handles, loc='upper left', frameon=True, facecolor='white', edgecolor='black', labelspacing=1.2)
    plt.tight_layout()
    
    # =========================================================
    # 5. EXPORT
    # =========================================================
    # save_path = os.path.join(output_dir, f"Perspective_Stratified_Phi_{attribute_key.replace('/', '_')}_{title_event_label}.png")
    # plt.savefig(save_path, facecolor='white', edgecolor='none')
    # if show_plot.lower() == 'y': plt.show()
    # else: plt.close(fig)
    
    
    # plt.show()
    return fig





def t_phi_attr_R_3d(mother_dict, attribute_key, event_type, silso_csv):
    """
    Generates a true 3D spatial visualization of ICME events.
    X: Time | Y: Longitude (φ) | Z: Radial Distance.
    Size: Fixed | Color: Attribute Value.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    target_keys = []
    if event_type.lower() in ['me', 'icme', 'both']: target_keys.append('ME')
    if event_type.lower() in ['sheath', 'icme', 'both']: target_keys.append('Sheath')
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()

    print(f"\n--- Constructing 3D Spatial Topology (Longitude) for {attribute_key} ({title_event_label}) ---")

    # =========================================================
    # 1. EVENT-LEVEL DATA EXTRACTION
    # =========================================================
    records = []
    
    for event_id, event_data in mother_dict.items():
        # 1. Extract base Year from the dictionary key (e.g., 'HL10001_1979')
        match_year = re.search(r'(19|20)\d{2}', event_id)
        if not match_year:
            continue
        year = float(match_year.group(0))

        # 2. FETCH R DIRECTLY FROM METADATA
        metadata = event_data.get('Metadata', {})
        event_R = metadata.get('r_AU') 
        
        if event_R is None:
            continue

        event_attr_pool, event_phi_pool, event_doy_pool = [], [], []

        for key in target_keys:
            df = event_data.get(key)
            if df is None or attribute_key not in df.columns or 'Location' not in df.columns:
                continue

            # 3. Extract precise DOY from Epoch_ID (e.g., 'HL10001_1979_305.039259')
            if 'Epoch_ID' in df.columns:
                for epoch_val in df['Epoch_ID'].dropna():
                    match_doy = re.search(r'_(\d+\.\d+)$', str(epoch_val))
                    if match_doy:
                        event_doy_pool.append(float(match_doy.group(1)))

            # Extract Attributes
            raw_vals = pd.to_numeric(df[attribute_key], errors='coerce').dropna().tolist()
            event_attr_pool.extend(raw_vals)

            # Extract Spatial Vectors (Index 1 for Phi)
            for loc in df['Location'].dropna():
                loc_str = str(loc).replace('nan', 'None')
                try:
                    t = ast.literal_eval(loc_str)
                    # Index 1 corresponds to Longitude (Phi)
                    if t[1] is not None:
                        event_phi_pool.append(float(t[1]))
                except:
                    pass
        
        # 4. Compile mathematical mean for the event
        if event_attr_pool and event_phi_pool:
            
            # Calculate precise decimal year using DOY pool
            if event_doy_pool:
                mean_doy = np.mean(event_doy_pool)
                dec_year = year + (mean_doy / 365.25)
            else:
                print(f"Warning: No valid DOY found in Epoch_ID for {event_id}. Skipping.")
                continue 

            records.append({
                'Decimal_Year': dec_year,
                'R': float(event_R),
                'Phi': np.mean(event_phi_pool),
                'Attr_Mean': np.mean(event_attr_pool)
            })

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        print("CRITICAL: No valid spatial/attribute data found for this combination.")
        return

    # =========================================================
    # 2. 3D VISUALIZATION ARCHITECTURE
    # =========================================================
    fig = plt.figure(figsize=(16, 10), dpi=150)
    ax1 = fig.add_subplot(111, projection='3d') 

    min_yr = np.floor(df_plot['Decimal_Year'].min())
    max_yr = np.ceil(df_plot['Decimal_Year'].max())
    max_phi = max(abs(df_plot['Phi'])) + 10 # Updated from max_lat
    max_R_plot = df_plot['R'].max()

    # --- Background: SILSO Solar Cycle (Projected on Z=0 Plane) ---
    try:
        df_ssn = load_local_silso_data(silso_csv)
        if not df_ssn.empty:
            mask = (df_ssn['Decimal_Year'] >= min_yr - 1) & (df_ssn['Decimal_Year'] <= max_yr + 1)
            silso_clipped = df_ssn[mask]
            
            # Scale SSN amplitude to fit within Longitude bounds
            max_ssn = silso_clipped['SSN'].max()
            scaled_ssn = -max_phi + (silso_clipped['SSN'] / max_ssn) * (max_phi * 2)
            
            ax1.plot(silso_clipped['Decimal_Year'], scaled_ssn, zs=0, zdir='z', 
                     color='gray', alpha=0.3, linewidth=2, label='Scaled Solar Cycle')
    except Exception as e:
        print(f"Warning: Could not load SILSO background ({e})")

    # --- Solar Equator (The Sun) ---
    ax1.plot([min_yr, max_yr], [0, 0], zs=0, zdir='z', color='gold', linewidth=3, alpha=0.8, label='Solar Equator (Sun)')

    # --- Foreground: The 3D Scatter (Fixed Size) ---
    scatter = ax1.scatter(
        df_plot['Decimal_Year'], 
        df_plot['Phi'], 
        df_plot['R'],           
        c=df_plot['Attr_Mean'], 
        s=40,                   
        cmap='viridis_r',          
        alpha=0.85,          
        edgecolors='black', 
        linewidth=0.5,
        depthshade=True         
    )

    # =========================================================
    # 3. FORMATTING & AESTHETICS
    # =========================================================
    ax1.set_title(f"Solar Cycle, Longitude, R, and {attribute_key} ({title_event_label})", 
                  fontweight='bold', fontsize=16, pad=20)
    
    ax1.set_xlabel("Time [Years]", fontweight='bold', fontsize=11, labelpad=10)
    ax1.set_ylabel("Longitude (φ) [Degrees]", fontweight='bold', fontsize=11, labelpad=10) # Updated Label
    ax1.set_zlabel("Radial Distance (R) [AU]", fontweight='bold', fontsize=11, labelpad=10)
    
    ax1.set_ylim(-max_phi, max_phi) # Updated from max_lat
    ax1.set_zlim(0, max_R_plot * 1.05) 
    
    ax1.set_xlim(min_yr, max_yr)
    ax1.set_xticks(np.arange(min_yr, max_yr + 1, 3)) 
    ax1.tick_params(axis='x', rotation=45)

    ax1.view_init(elev=20, azim=-60)

    # --- Colorbar ---
    cbar = fig.colorbar(scatter, ax=ax1, pad=0.05, shrink=0.7)
    cbar.set_label(f'Mean {attribute_key} Value', fontweight='bold', fontsize=11)
    
    # --- Custom Legend ---
    custom_handles = [
        Line2D([0], [0], color='gold', linewidth=3, label='Solar Equator'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.4, label='Solar Cycle Timeline')
    ]
    
    ax1.legend(handles=custom_handles, loc='upper left', 
               bbox_to_anchor=(0.0, 1.0), frameon=True, facecolor='white', edgecolor='black', labelspacing=1.2)

    plt.tight_layout()
    
    # =========================================================
    # 4. EXPORT
    # =========================================================
    safe_filename = attribute_key.replace("/", "_")
    # save_path = os.path.join(output_dir, f"Perspective_3D_Phi_{safe_filename}_{title_event_label}.png")
    # plt.savefig(save_path, facecolor='white', edgecolor='none')
    # print(f"Plot successfully saved to: {save_path}")
    
    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)
    
    
    # plt.show()
    return fig





def attr_events(mother_dict, event_type, radial_bin, attribute_key):
    """
    Generates a Histogram (Raw Occurrences vs Attribute Value) for a requested parameter.
    
    Arguments:
    event_type    : 'ME', 'Sheath', or 'Both' (or 'ICME')
    radial_bin    : tuple of floats, e.g., (0.8, 1.2)
    attribute_key : e.g., 'beta', 'B', 'Vsw'
    show_plot     : 'y' to display the plot inline, any other string to just save and close.
    """
    # 1. INPUT VALIDATION
    if attribute_key not in ATTRIBUTE_LABELS:
        print(f"CRITICAL: '{attribute_key}' is not a valid attribute key.")
        return
        
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
        
    axis_label = ATTRIBUTE_LABELS[attribute_key]
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()
    r_min, r_max = radial_bin
    
    print(f"\n--- Calculating Counts for '{attribute_key}' ({title_event_label}) in {r_min} - {r_max} AU ---")
    
    # 2. TENSOR AGGREGATION
    aggregated_me = []
    aggregated_sheath = []
    
    for event_id, event_data in mother_dict.items():
        r_val = event_data.get('Metadata', {}).get('r_AU', None)
        
        if r_val is None or not (r_min <= r_val <= r_max):
            continue
            
        if event_type.lower() in ['me', 'both', 'icme'] and event_data.get('ME') is not None:
            aggregated_me.append(event_data['ME'])
            
        if event_type.lower() in ['sheath', 'both', 'icme'] and event_data.get('Sheath') is not None:
            aggregated_sheath.append(event_data['Sheath'])

    if not aggregated_me and not aggregated_sheath:
        print(f"CRITICAL: No valid data found in the {r_min} - {r_max} AU bin.")
        return

    # Master DataFrames
    df_me = pd.concat(aggregated_me, ignore_index=True) if aggregated_me else pd.DataFrame()
    df_sh = pd.concat(aggregated_sheath, ignore_index=True) if aggregated_sheath else pd.DataFrame()

    # 3. EXTRACT SPECIFIC ARRAY
    array_me = pd.to_numeric(df_me[attribute_key], errors='coerce').dropna() if attribute_key in df_me.columns else pd.Series(dtype=float)
    array_sh = pd.to_numeric(df_sh[attribute_key], errors='coerce').dropna() if attribute_key in df_sh.columns else pd.Series(dtype=float)

    if array_me.empty and array_sh.empty:
        print(f"CRITICAL: The column '{attribute_key}' contains no valid numeric data in this radial bin.")
        return

    # 4. VISUALIZATION ARCHITECTURE
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Use step-histograms for clean overlapping boundaries
    if not array_me.empty and event_type.lower() in ['me', 'both', 'icme']:
        sns.histplot(array_me, color="blue", label="ME", ax=ax, bins = 100, alpha=0.3, stat='count', element='step', lw=2)
        
    if not array_sh.empty and event_type.lower() in ['sheath', 'both', 'icme']:
        sns.histplot(array_sh, color="red", label="Sheath", ax=ax, bins = 100, alpha=0.3, stat='count', element='step', lw=2)

    # 5. FORMATTING
    ax.set_xlabel(f"{axis_label}", fontweight='bold', fontsize=12)
    ax.set_ylabel("Events", fontweight='bold', fontsize=12)
    title_str = f"Event Count of {axis_label}\n {title_event_label} ({r_min} - {r_max} AU)"
    ax.set_title(title_str, fontweight='bold', fontsize=13, pad=15)
    
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 6. EXPORT & MEMORY MANAGEMENT
    plt.tight_layout()
    safe_filename = attribute_key.replace("/", "_")
    # save_path = os.path.join(output_dir, f"Count_{safe_filename}_{title_event_label}_{r_min}_{r_max}AU.png")
    # plt.savefig(save_path)
    
    # print(f"Plot successfully saved to: {save_path}")
    
    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)
    
    
    # plt.show()
    return fig

# =========================================================
# Execution Example:
# plot_event_counts(mother_dict, event_type="Both", radial_bin=(0.8, 1.2), attribute_key="B", show_plot='n')
# attribute_key: 
#   'B'           (Magnetic Field Intensity)
#   'Np'          (Proton Density)
#   'Vsw'         (Bulk velocity)
#   'Tp'          (Proton Temperature)
#   'Va'          (Alfvén Speed)
#   'M_A'         (Alfvén Mach Number)
#   'beta'        (Plasma Beta)
#   'P_dyn'       (Dynamic Pressure)
#   'P_T'         (Total Pressure)
#   'T_r'         (Temperature ratio)
#   'S'           (Entropy)
#   'Na/Np'       (Alpha to proton ratio)
#   'rmsB_over_B' (Normalized magnetic field fluctuation amplitude)
#   'deltaB'      (Fluctuation in B)
# show_plot : 'y' or anything else





def R_phi_R_theta(mother_dict, event_type):

    # 1. EVENT ROUTING LOGIC
    # Decouples the user's string from the strict dictionary keys
    target_keys = []
    if event_type.lower() in ['me', 'icme', 'both']:
        target_keys.append('ME')
    if event_type.lower() in ['sheath', 'icme', 'both']:
        target_keys.append('Sheath')
        
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()
    print(f"Aggregating spatial coordinates for {title_event_label}...")
    
    # 2. EXTRACT SPATIAL TENSORS
    r_vals, phi_vals, theta_vals = [], [], []
    
    for event_id, event_data in mother_dict.items():
        # Iterate through the routed keys to capture all requested data
        for key in target_keys:
            if event_data.get(key) is None:
                continue
                
            df = event_data[key]
            if 'Location' not in df.columns:
                continue
                
            # Parse the location tuples (R, Phi, Theta)
            for loc in df['Location'].dropna():
                loc_str = str(loc).replace('nan', 'None')
                try:
                    t = ast.literal_eval(loc_str)
                    if t[0] is not None and t[1] is not None and t[2] is not None:
                        r_vals.append(float(t[0]))
                        phi_vals.append(float(t[1]))
                        theta_vals.append(float(t[2]))
                except:
                    pass
                
    if not r_vals:
        print("CRITICAL: No valid spatial data found.")
        return

    # Convert to NumPy arrays
    R = np.array(r_vals)
    Phi_rad = np.radians(np.array(phi_vals)) 
    Theta = np.array(theta_vals)

    # 3. VISUALIZATION ARCHITECTURE
    fig = plt.figure(figsize=(14, 6), dpi=150)
    
    # --- PANEL 1: Ecliptic Plane (Polar Heatmap) ---
    ax1 = fig.add_subplot(121, projection='polar')
    
    phi_bins = np.linspace(0, 2 * np.pi, 36) 
    r_bins = np.linspace(0, max(R) + 0.5, 20)
    
    H_polar, phi_edges, r_edges = np.histogram2d(Phi_rad, R, bins=[phi_bins, r_bins])
    
    Phi_mesh, R_mesh = np.meshgrid(phi_edges, r_edges)
    cmap_polar = ax1.pcolormesh(Phi_mesh, R_mesh, H_polar.T, 
                                cmap='inferno', 
                                norm=mcolors.LogNorm(vmin=1, vmax=H_polar.max() + 1))
    
    ax1.set_title("Longitude vs Radial Distance", fontweight='bold', pad=15)
    cbar1 = fig.colorbar(cmap_polar, ax=ax1, fraction=0.046, pad=0.08)
    cbar1.set_label('Events (Log Scale)', fontweight='bold')
    
    # --- PANEL 2: Meridional Plane (Cartesian Heatmap) ---
    ax2 = fig.add_subplot(122)
    
    hb = ax2.hexbin(R, Theta, gridsize=30, cmap='inferno', 
                    bins='log', mincnt=1, edgecolors='none')
    
    ax2.set_title("Latitude vs Radial Distance", fontweight='bold', pad=15)
    ax2.set_xlabel("Radial Distance (R) [AU]", fontweight='bold')
    ax2.set_ylabel("Latitude (θ) [Degrees]", fontweight='bold')
    
    max_lat = max(abs(Theta)) + 5
    ax2.set_ylim(-max_lat, max_lat)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    cbar2 = fig.colorbar(hb, ax=ax2)
    cbar2.set_label('Events (Log Scale)', fontweight='bold')

    plt.suptitle(f"Spatial Distribution - {title_event_label}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(w_pad=4.0)
    
    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)


    # plt.show()
    return fig
# Execution Example:
# plot_spatial_icme_density(mother_dict, event_type='ME', show_plot = 'y')





def spatial_events_4D(mother_dict, event_type, phi_step=15, theta_step=5):
    """
    Plots a 4D Heliographic Sky Map on the Phi-Theta plane.
    Marker Color = Event Count (Log Scaled).
    Marker Size = Inverse of Mean Radial Distance (Bigger = Closer to Sun).
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    print(f"Aggregating 4D spatial tensors for {event_type}...")
    
    # 1. EXTRACT SPATIAL TENSORS
    records = []
    
    for event_id, event_data in mother_dict.items():
        if event_data.get(event_type) is None:
            continue
            
        df = event_data[event_type]
        if 'Location' not in df.columns:
            continue
            
        for loc in df['Location'].dropna():
            loc_str = str(loc).replace('nan', 'None')
            try:
                t = ast.literal_eval(loc_str)
                if t[0] is not None and t[1] is not None and t[2] is not None:
                    records.append({'R': float(t[0]), 'Phi': float(t[1]), 'Theta': float(t[2])})
            except:
                pass
                
    if not records:
        print("CRITICAL: No valid spatial data found.")
        return

    df_plot = pd.DataFrame(records)

    # 2. THE 2D ANGULAR BINNING ENGINE
    phi_edges = np.arange(0, 360 + phi_step, phi_step)
    theta_edges = np.arange(-90, 90 + theta_step, theta_step)

    df_plot['Phi_bin'] = pd.cut(df_plot['Phi'], bins=phi_edges)
    df_plot['Theta_bin'] = pd.cut(df_plot['Theta'], bins=theta_edges)

    agg_df = df_plot.groupby(['Phi_bin', 'Theta_bin'], observed=True).agg(
        Count=('R', 'size'),
        Mean_R=('R', 'mean')
    ).reset_index()

    agg_df = agg_df[agg_df['Count'] > 0].copy()

    # =========================================================
    # THE FIX: Explicitly cast the categorical midpoints to floats
    # =========================================================
    agg_df['Phi_mid'] = agg_df['Phi_bin'].apply(lambda x: x.mid).astype(float)
    agg_df['Theta_mid'] = agg_df['Theta_bin'].apply(lambda x: x.mid).astype(float)

    # 3. DEPTH-OF-FIELD SCALING
    base_marker_size = 300 
    sizes = base_marker_size / agg_df['Mean_R']

    # 4. VISUALIZATION ARCHITECTURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    scatter = ax.scatter(
        agg_df['Phi_mid'], 
        agg_df['Theta_mid'], 
        s=sizes, 
        c=agg_df['Count'], 
        cmap='plasma', 
        norm=mcolors.LogNorm(vmin=1, vmax=agg_df['Count'].max()), 
        alpha=0.85, 
        edgecolors='black', 
        linewidth=0.5
    )

    # 5. FORMATTING & AESTHETICS
    ax.set_title(f"Spatial Event Distribution: {event_type.upper()}", fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel("Longitude (φ) [Degrees]", fontweight='bold', fontsize=12)
    ax.set_ylabel("Latitude (θ) [Degrees]", fontweight='bold', fontsize=12)
    
    ax.set_xlim(0, 360)
    
    # This line will now execute perfectly because Theta_mid is a pure float
    max_lat = max(abs(agg_df['Theta_mid'])) + theta_step
    ax.set_ylim(-max_lat, max_lat)
    
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Events (Log Scale)', fontweight='bold', fontsize=11) # Ensure this is imported at the top of your script

    # Generate the raw Matplotlib handles and LaTeX-formatted labels
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4, func=lambda s: base_marker_size / s)
    
    clean_labels = []
    for l in labels:
        # Regex safely hunts down the floating-point number hidden inside the string
        match = re.search(r"[-+]?\d*\.\d+|\d+", l)
        if match:
            clean_val = float(match.group())
            clean_labels.append(f"{clean_val:.1f} AU")
        else:
            clean_labels.append(l) # Failsafe
            
    # Apply the sanitized labels to the legend
    legend = ax.legend(handles, clean_labels, 
                       title="Mean Distance (R)", loc="upper right", framealpha=1)
    
    plt.tight_layout()
    
    # 6. EXPORT & DISPLAY
    # save_path = os.path.join(output_dir, f"SkyMap_{event_type.upper()}.png")
    # plt.savefig(save_path)
    
    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)



    # plt.show()
    return fig
# =========================================================
# Execution Example:
# plot_heliospheric_sky_map(mother_dict_all, event_type='ME', show_plot='y')





def spacecraft_events(mother_dict, event_type):
    """
    Plots a sorted bar chart showing the total number of distinct plasma regions 
    (ME / Sheath) detected by each individual spacecraft.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Clarify the title to reflect the additive nature of the data
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()
    print(f"\n--- Aggregating Spacecraft Demographics for {title_event_label} ---")

    # =========================================================
    # 1. METADATA EXTRACTION (Additive Logic)
    # =========================================================
    spacecraft_counts = {}

    for event_id, event_data in mother_dict.items():
        has_me = event_data.get('ME') is not None
        has_sheath = event_data.get('Sheath') is not None
        
        # Shift from a boolean flag to an integer multiplier
        regions_to_count = 0
        
        if event_type.lower() == 'me' and has_me:
            regions_to_count = 1
        elif event_type.lower() == 'sheath' and has_sheath:
            regions_to_count = 1
        elif event_type.lower() in ['icme', 'both']:
            # Additively count both regions if they exist in the dictionary
            if has_me: 
                regions_to_count += 1
            if has_sheath: 
                regions_to_count += 1
                
        # Log the total regions found for this specific event to the spacecraft
        if regions_to_count > 0:
            sc = event_data.get('Metadata', {}).get('spacecraft', 'Unknown')
            spacecraft_counts[sc] = spacecraft_counts.get(sc, 0) + regions_to_count

    if not spacecraft_counts:
        print("CRITICAL: No valid spacecraft data found.")
        return

    # Convert to DataFrame and sort from highest to lowest count
    df_counts = pd.DataFrame(list(spacecraft_counts.items()), columns=['Spacecraft', 'Count'])
    df_counts = df_counts.sort_values(by='Count', ascending=False)

    # =========================================================
    # 2. VISUALIZATION ARCHITECTURE
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot the bars
    sns.barplot(x='Spacecraft', y='Count', data=df_counts, palette='viridis', ax=ax, edgecolor='black')

    # =========================================================
    # 3. FORMATTING & AESTHETICS
    # =========================================================
    ax.set_title(f"Total {title_event_label} Events Detected per Spacecraft", fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel("Spacecraft", fontweight='bold', fontsize=12)
    
    # Critically updated Y-axis label to reflect computational regions, not physical events
    ax.set_ylabel("Events", fontweight='bold', fontsize=12)

    # Dynamically annotate the exact count on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', 
                    fontsize=11, fontweight='bold', color='black', 
                    xytext=(0, 5), textcoords='offset points')

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # =========================================================
    # 4. EXPORT & MEMORY MANAGEMENT
    # =========================================================
    # save_path = os.path.join(output_dir, f"Spacecraft_Count_{event_type.upper()}.png")
    # plt.savefig(save_path)
    # print(f"Plot successfully saved to: {save_path}")

    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)



    # plt.show()
    return fig
# =========================================================
# Execution Example:
# plot_spacecraft_event_counts(mother_dict, event_type='Both', show_plot='y')





def events_over_butterfly(mother_dict, event_type, df_rgo):
    """
    Generates a classical Solar Cycle Butterfly Diagram for ICMEs.
    X-axis: Decimal Year.
    Y-axis: Latitude (Theta).
    Color: Radial Distance (R).
    Background: RGO/USAF/NOAA Sunspot Butterfly Diagram.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # 1. EVENT ROUTING LOGIC
    target_keys = []
    if event_type.lower() in ['me', 'icme', 'both']:
        target_keys.append('ME')
    if event_type.lower() in ['sheath', 'icme', 'both']:
        target_keys.append('Sheath')
        
    title_event_label = "ICME" if event_type.lower() in ["both", "icme"] else event_type.upper()

    print(f"\n--- Constructing Fused Butterfly Diagram for {title_event_label} ---")
    
    # =========================================================
    # 2. TEMPORAL & SPATIAL DATA EXTRACTION (Corrected Timeline)
    # =========================================================
    records = []
    
    for event_id, event_data in mother_dict.items():
        # 1. Extract base Year strictly from the dictionary key
        match_year = re.search(r'(19|20)\d{2}', event_id)
        if not match_year:
            continue
        year = float(match_year.group(0))

        # Iterate through the requested regions (ME, Sheath, or Both)
        for key in target_keys:
            df = event_data.get(key)
            if df is None or 'Location' not in df.columns or 'Epoch_ID' not in df.columns:
                continue

            # 2. Extract precise DOY and Spatial Data row-by-row for max fidelity
            for _, row in df.iterrows():
                loc = row['Location']
                epoch_val = row['Epoch_ID']
                
                if pd.isna(loc) or pd.isna(epoch_val):
                    continue
                
                # Extract precise DOY from this specific row's Epoch_ID
                match_doy = re.search(r'_(\d+\.\d+)$', str(epoch_val))
                if not match_doy:
                    continue
                    
                dec_year = year + (float(match_doy.group(1)) / 365.25)

                # Extract Spatial Tuples (R, Phi, Theta)
                loc_str = str(loc).replace('nan', 'None')
                try:
                    t = ast.literal_eval(loc_str)
                    # Index 0 is R, Index 2 is Theta
                    if t[0] is not None and t[2] is not None:
                        records.append({
                            'Decimal_Year': dec_year, 
                            'R': float(t[0]), 
                            'Theta': float(t[2])
                        })
                except (ValueError, SyntaxError, TypeError):
                    pass
                
    if not records:
        print("CRITICAL: No valid spatio-temporal data found in the dictionary.")
        return

    df_plot = pd.DataFrame(records)

    # =========================================================
    # 3. RGO SUNSPOT INGESTION (BACKGROUND BUTTERFLY)
    # =========================================================
    # try:
    #     df_rgo = ingest_hathaway_rgo_data(rgo_path)
    # except NameError:
    #     print("Warning: 'ingest_hathaway_rgo_data' function not defined in this scope. Bypassing RGO background.")
    #     df_rgo = pd.DataFrame()

    # =========================================================
    # 4. VISUALIZATION ARCHITECTURE
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)
    
    min_yr = np.floor(df_plot['Decimal_Year'].min())
    max_yr = np.ceil(df_plot['Decimal_Year'].max())

    # --- The RGO Sunspot Background Watermark (Refined) ---
    if not df_rgo.empty:
        mask_rgo = (df_rgo['Decimal_Year'] >= min_yr - 1) & (df_rgo['Decimal_Year'] <= max_yr + 1)
        rgo_clipped = df_rgo[mask_rgo]
        
        marker_sizes = rgo_clipped['Area'] / 35.0 

        ax1.scatter(
            rgo_clipped['Decimal_Year'], 
            rgo_clipped['Latitude'], 
            color='cornflowerblue', 
            s=marker_sizes, 
            alpha=0.2,            
            edgecolors='none', 
            zorder=1, 
            label='Sunspots',
            rasterized=True       
        )

    # --- The Primary Butterfly Scatter (ICMEs) ---
    scatter = ax1.scatter(
        df_plot['Decimal_Year'], 
        df_plot['Theta'], 
        c=df_plot['R'], 
        cmap='viridis', 
        s=40,          
        alpha=0.8, 
        edgecolors='black', 
        linewidth=0.5,
        zorder=3       
    )

    # =========================================================
    # 5. FORMATTING & AESTHETICS
    # =========================================================
    ax1.set_title(f"Event Distribution: {title_event_label} and Sunspot Latitudinal Evolution", 
                  fontweight='bold', fontsize=16, pad=15)
    
    ax1.set_xlabel("Time [Years]", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Latitude (θ) [Degrees]", fontweight='bold', fontsize=12)
    
    icme_max_lat = max(abs(df_plot['Theta']))
    rgo_max_lat = max(abs(rgo_clipped['Latitude'])) if not df_rgo.empty else 0
    max_lat = max(icme_max_lat, rgo_max_lat) + 5
    ax1.set_ylim(-max_lat, max_lat)
    
    ax1.set_xlim(min_yr, max_yr)
    ax1.set_xticks(np.arange(min_yr, max_yr + 1, 1))
    ax1.tick_params(axis='x', rotation=45)
    
    ax1.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6, zorder=2)
    ax1.grid(True, linestyle='--', alpha=0.4, zorder=0)

    cbar = fig.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('Radial Distance (R) [AU]', fontweight='bold', fontsize=11)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='upper right', frameon=True, edgecolor='black', markerscale=3)

    plt.tight_layout()
    
    # =========================================================
    # 6. EXPORT & MEMORY MANAGEMENT
    # =========================================================
    # save_path = os.path.join(output_dir, f"Fused_Butterfly_{title_event_label}.png")
    # plt.savefig(save_path)
    # print(f"Plot successfully saved to: {save_path}")
    
    # if show_plot.lower() == 'y':
    #     plt.show()
    # else:
    #     plt.close(fig)
    
    
    # plt.show()
    return fig





