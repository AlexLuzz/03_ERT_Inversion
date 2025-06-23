import pandas as pd
import xml.etree.ElementTree as ET
import os
from datetime import datetime

def xle_to_csv(xle_file_path, output_csv_path=None):
    """
    Convert XLE file (XML Logger Exchange) to CSV format
    Specifically designed for LTC data format
    
    Parameters:
    -----------
    xle_file_path : str
        Path to the input .xle file
    output_csv_path : str, optional
        Path for output CSV file. If None, uses same name as input with .csv extension
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the converted data
    """
    
    # Generate output path if not provided
    if output_csv_path is None:
        base_name = os.path.splitext(xle_file_path)[0]
        output_csv_path = f"{base_name}.csv"
    
    try:
        # Parse the XML file
        tree = ET.parse(xle_file_path)
        root = tree.getroot()
        
        # Extract channel information for proper column naming
        ch1_info = root.find('Ch1_data_header')
        ch2_info = root.find('Ch2_data_header')
        
        ch1_name = "ch1"
        ch1_unit = ""
        ch2_name = "ch2" 
        ch2_unit = ""
        
        if ch1_info is not None:
            ch1_id = ch1_info.find('Identification')
            ch1_unit_elem = ch1_info.find('Unit')
            if ch1_id is not None:
                ch1_name = ch1_id.text.strip()
            if ch1_unit_elem is not None:
                ch1_unit = ch1_unit_elem.text.strip()
        
        if ch2_info is not None:
            ch2_id = ch2_info.find('Identification')
            ch2_unit_elem = ch2_info.find('Unit')
            if ch2_id is not None:
                ch2_name = ch2_id.text.strip()
            if ch2_unit_elem is not None:
                ch2_unit = ch2_unit_elem.text.strip()
        
        # Create proper column names with units
        ch1_col = f"{ch1_name} ({ch1_unit})" if ch1_unit else ch1_name
        ch2_col = f"{ch2_name} ({ch2_unit})" if ch2_unit else ch2_name
        
        print(f"Channel 1: {ch1_col}")
        print(f"Channel 2: {ch2_col}")
        
        # Find the data section and extract all Log entries
        data_records = []
        
        data_section = root.find('Data')
        if data_section is not None:
            for log in data_section.findall('Log'):
                record_data = {}
                
                # Extract date, time, and ms
                date_elem = log.find('Date')
                time_elem = log.find('Time')
                ms_elem = log.find('ms')
                
                if date_elem is not None and time_elem is not None:
                    date_str = date_elem.text.strip()
                    time_str = time_elem.text.strip()
                    ms_str = ms_elem.text.strip() if ms_elem is not None else "0"
                    
                    # Create full datetime string
                    datetime_str = f"{date_str} {time_str}"
                    record_data['DateTime'] = datetime_str
                    record_data['Milliseconds'] = int(ms_str)
                
                # Extract channel data
                ch1_elem = log.find('ch1')
                ch2_elem = log.find('ch2')
                
                if ch1_elem is not None and ch1_elem.text:
                    try:
                        record_data[ch1_col] = float(ch1_elem.text.strip())
                    except ValueError:
                        record_data[ch1_col] = ch1_elem.text.strip()
                
                if ch2_elem is not None and ch2_elem.text:
                    try:
                        record_data[ch2_col] = float(ch2_elem.text.strip())
                    except ValueError:
                        record_data[ch2_col] = ch2_elem.text.strip()
                
                # Add log ID if present
                log_id = log.get('id')
                if log_id:
                    record_data['Log_ID'] = int(log_id)
                
                if record_data:
                    data_records.append(record_data)
        
        # Create DataFrame
        if data_records:
            df = pd.DataFrame(data_records)
            
            # Convert DateTime column to proper datetime
            if 'DateTime' in df.columns:
                try:
                    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y/%m/%d %H:%M:%S')
                except:
                    print("Warning: Could not parse DateTime column with standard format")
                    try:
                        df['DateTime'] = pd.to_datetime(df['DateTime'])
                    except:
                        print("Warning: Could not parse DateTime column at all")
            
            # Reorder columns for better readability
            column_order = ['DateTime', 'Log_ID', 'Milliseconds', ch1_col, ch2_col]
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # Save to CSV
            df.to_csv(output_csv_path, index=False, sep=';')
            
            print(f"Successfully converted {xle_file_path} to {output_csv_path}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            print(f"Columns: {list(df.columns)}")
            
            # Show basic statistics
            if ch1_col in df.columns:
                print(f"{ch1_col} range: {df[ch1_col].min():.3f} to {df[ch1_col].max():.3f}")
            if ch2_col in df.columns:
                print(f"{ch2_col} range: {df[ch2_col].min():.3f} to {df[ch2_col].max():.3f}")
            
            return df
        
        else:
            print("No data records found in the XLE file")
            return None
            
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"Error converting file: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_instrument_info(xle_file_path):
    """
    Extract instrument and project information from XLE file
    
    Parameters:
    -----------
    xle_file_path : str
        Path to the input .xle file
        
    Returns:
    --------
    dict
        Dictionary containing instrument information
    """
    
    try:
        tree = ET.parse(xle_file_path)
        root = tree.getroot()
        
        info = {}
        
        # File info
        file_info = root.find('File_info')
        if file_info is not None:
            info['file_info'] = {}
            for child in file_info:
                info['file_info'][child.tag] = child.text
        
        # Instrument info
        instrument_info = root.find('Instrument_info')
        if instrument_info is not None:
            info['instrument_info'] = {}
            for child in instrument_info:
                info['instrument_info'][child.tag] = child.text
        
        # Instrument data header
        data_header = root.find('Instrument_info_data_header')
        if data_header is not None:
            info['data_header'] = {}
            for child in data_header:
                info['data_header'][child.tag] = child.text
        
        return info
        
    except Exception as e:
        print(f"Error extracting instrument info: {e}")
        return None

def convert_multiple_xle_files(folder_path, output_folder=None):
    """
    Convert all XLE files in a folder to CSV
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing .xle files
    output_folder : str, optional
        Output folder for CSV files. If None, uses same folder as input
    """
    
    if output_folder is None:
        output_folder = folder_path
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all XLE files
    xle_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xle')]
    
    if not xle_files:
        print(f"No .xle files found in {folder_path}")
        return
    
    print(f"Found {len(xle_files)} XLE files to convert...")
    
    converted_files = []
    
    for xle_file in xle_files:
        input_path = os.path.join(folder_path, xle_file)
        output_path = os.path.join(output_folder, xle_file.replace('.xle', '.csv'))
        
        print(f"\nConverting: {xle_file}")
        df = xle_to_csv(input_path, output_path)
        
        if df is not None:
            print(f"✓ Success: {len(df)} rows converted")
            converted_files.append(output_path)
        else:
            print(f"✗ Failed to convert {xle_file}")
    
    print(f"\n{len(converted_files)} files successfully converted!")
    return converted_files

if __name__ == "__main__":
    
    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_ETS

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/General - Projet IV 2023 - GTO365/01-projet_IV-Mtl_Laval/03-Berlier-Bergman/05-donnees-terrains/'

    # Convert a single file
    xle_file = Onedrive_path + "BB-BARO_20250515.xle"
    df = xle_to_csv(xle_file)
    
    # Convert all XLE files in a folder
    # folder_path = "path/to/your/xle/files/"
    # convert_multiple_xle_files(folder_path)
    
    if os.path.exists(xle_file):
        df = xle_to_csv(xle_file)
        if df is not None:
            print("\nFirst few rows of converted data:")
            print(df.head())
    else:
        print(f"File {xle_file} not found. Please update the file path.")