import os
import shutil

# Create the .kaggle directory if it doesn't exist
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Move kaggle.json to the .kaggle directory
shutil.move("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))

# Set the file permissions to read-only
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# command :
#  kaggle datasets download -d jangedoo/utkface-new -p data/utkface --unzip  : download in data/utkface folder & unzip it
for root, dirs, files in os.walk("data/utkface"):
    for file in files:
        print(file)