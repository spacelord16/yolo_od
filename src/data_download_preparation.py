import os 
import shutil

def setup_kaggle():
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy('../kaggle.json', f'{kaggle_dir}/kaggle.json')
    os.chmod(f'{kaggle_dir}/kaggle.json', 0o600)

def download_dataset():
    setup_kaggle()
    data_dir = '../data/raw'
    os.makedirs(data_dir, exist_ok=True)
    
    # clearer command output
    exit_code = os.system(f'kaggle datasets download -d huanghanchina/pascal-voc-2012 -p {data_dir}')
    
    if exit_code != 0:
        raise Exception("Dataset download failed. Check your Kaggle credentials and dataset path.")

    zip_file = os.path.join(data_dir, 'pascal-voc-2012.zip')

    if not os.path.exists(zip_file):
        raise FileNotFoundError(f"{zip_file} was not downloaded successfully.")

    shutil.unpack_archive(zip_file, data_dir)
    print("Dataset downloaded and extracted successfully.")

if __name__ == "__main__":
    download_dataset()
