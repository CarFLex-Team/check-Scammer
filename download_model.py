import gdown
import zipfile
import os

FILE_ID = "1ildQzqDnYel42zSkki6EW1c9scg3BT59"
ZIP_NAME = "bart_mnli_model.zip"
MODEL_DIR = "bart_mnli_model"

if not os.path.exists(MODEL_DIR):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_NAME, quiet=False)

    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(".")

    os.remove(ZIP_NAME)
    print("model downloaded")
else:
    print("model already exists")
