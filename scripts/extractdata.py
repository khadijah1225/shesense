import pickle
import pandas as pd
import sys
import os
import glob
import zipfile

def extract_wesad(input_pkl: str, output_csv: str):
    """
    Load WESAD pickle, extract HR, temperature, and labels, then save as CSV.
    """
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    df = pd.DataFrame({
        "timestamp": data['signal']['temp']['watch'].index,
        "hr": data['signal']['wrist']['BVP'].resample('1S').mean(),  # 1 Hz
        "temp": data['signal']['temp']['watch'],
        "label": data['label']
    }).dropna()
    df.to_csv(output_csv, index=False)
    print(f"WESAD data extracted to {output_csv}")

def extract_swell(raw_input: str, output_csv: str):
    """
    Load SWELL-KW dataset from a zip or directory, merge ECG and EDA per subject, then save as CSV.
    """
    # Determine if input is a zip; if so, extract
    raw_dir = raw_input.rstrip('.zip')
    if raw_input.lower().endswith('.zip') and os.path.isfile(raw_input):
        os.makedirs(raw_dir, exist_ok=True)
        with zipfile.ZipFile(raw_input, 'r') as z:
            z.extractall(raw_dir)
        print(f"Extracted zip to {raw_dir}")
    else:
        raw_dir = raw_input

    # Find ECG and EDA CSVs
    ecg_paths = glob.glob(os.path.join(raw_dir, '*ECG*.csv')) + glob.glob(os.path.join(raw_dir, '*ecg*.csv'))
    eda_paths = glob.glob(os.path.join(raw_dir, '*EDA*.csv')) + glob.glob(os.path.join(raw_dir, '*eda*.csv'))

    all_subjects = []
    for ecg_file in ecg_paths:
        subj_id = os.path.basename(ecg_file).split('.')[0]
        # match EDA by subject identifier
        eda_file = next((f for f in eda_paths if subj_id in os.path.basename(f)), None)

        ecg = pd.read_csv(ecg_file, parse_dates=['timestamp']).set_index('timestamp')
        if eda_file:
            eda = pd.read_csv(eda_file, parse_dates=['timestamp']).set_index('timestamp')
            df = ecg.join(eda, how='inner', rsuffix='_eda')
        else:
            df = ecg.copy()
        df = df.reset_index().dropna()
        df['subject'] = subj_id
        all_subjects.append(df)

    if not all_subjects:
        print("No SWELL files found in", raw_dir)
        return

    result = pd.concat(all_subjects, ignore_index=True)
    result.to_csv(output_csv, index=False)
    print(f"SWELL-KW data extracted to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract physiological datasets to CSV")
    parser.add_argument("dataset", choices=['wesad','swell'], help="Which dataset to extract")
    parser.add_argument("input", help="Path to WESAD pickle or SWELL zip/directory")
    parser.add_argument("output", help="Output CSV file path")
    args = parser.parse_args()

    if args.dataset == 'wesad':
        extract_wesad(args.input, args.output)
    else:
        extract_swell(args.input, args.output)
