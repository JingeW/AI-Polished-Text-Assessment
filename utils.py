import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_excel(file_path, sheet_name=None):
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        raise Exception(f"Failed to load excel file at {file_path}: {e}")

def save_to_txt(file_path, content):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
    except Exception as e:
        raise Exception(f"Failed to save file to {file_path}: {e}")

def save_to_json(metadata, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)
    except Exception as e:
        raise Exception(f"Failed to save metadata to {file_path}: {e}")

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at {file_path}.")
    except json.JSONDecodeError:
        raise ValueError(f"Metadata file at {file_path} is not a valid JSON file.")

def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Failed to load CSV file at {file_path}: {e}")

def save_to_csv(df, file_path, index=False):
    try:
        df.to_csv(file_path, index=index, encoding="utf-8")
    except Exception as e:
        raise Exception(f"Failed to save DataFrame to {file_path}: {e}")
    
def save_plot(filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), format="png", dpi=300)
    plt.savefig(os.path.join(output_dir, f"{filename}.svg"), format="svg")
    plt.close()
    print(f"Saved {filename}.png and {filename}.svg in {output_dir}.")