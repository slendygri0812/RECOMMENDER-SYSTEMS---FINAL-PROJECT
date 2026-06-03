import urllib.request
import zipfile
import os
import subprocess
import sys

print("Setting up Python 3.12 local environment...")

env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_env")
os.makedirs(env_dir, exist_ok=True)

# 1. Download Python 3.12.8 Embeddable
zip_path = os.path.join(env_dir, "python-3.12.8.zip")
url = "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip"
print(f"Downloading Python 3.12.8 Embeddable from {url}...")
urllib.request.urlretrieve(url, zip_path)
print("Downloaded successfully.")

# 2. Extract
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(env_dir)
os.remove(zip_path)
print("Extracted successfully.")

# 3. Configure _pth file
pth_path = os.path.join(env_dir, "python312._pth")
pth_content = """python312.zip
.

# Enable site packages
import site
"""
with open(pth_path, 'w') as f:
    f.write(pth_content)
print("Configured python312._pth successfully.")

# 4. Download get-pip.py
get_pip_path = os.path.join(env_dir, "get-pip.py")
pip_url = "https://bootstrap.pypa.io/get-pip.py"
print(f"Downloading get-pip.py from {pip_url}...")
urllib.request.urlretrieve(pip_url, get_pip_path)
print("Downloaded get-pip.py successfully.")

# 5. Install pip
python_exe = os.path.join(env_dir, "python.exe")
print("Installing pip...")
subprocess.run([python_exe, get_pip_path], check=True)
os.remove(get_pip_path)
print("Installed pip successfully.")

# 6. Install requirements
pip_exe = os.path.join(env_dir, "Scripts", "pip.exe")
print("Installing requirements (tensorflow, transformers, pandas, scikit-learn, tqdm)...")
subprocess.run([pip_exe, "install", "tensorflow", "transformers", "pandas", "scikit-learn", "tqdm"], check=True)

print("Python 3.12 and TensorFlow are successfully set up locally!")
