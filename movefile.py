import pandas as pd
import shutil 
import os

# Load the CSV file into a DataFrame
df = pd.read_csv("/media/data2/FakeAVCeleb_v1.2/FakeAVCeleb/meta_data.csv")

# Filter rows where the 'method' column contains the word 'real'
real_methods = df[df['method'].str.contains('real', na=False)]
no_real_methods = df[~df['method'].str.contains('real', na=False)]

shuffled_no_real_methods = no_real_methods.sample(frac=1).reset_index(drop=True)
for index, row in real_methods[:350].iterrows():
    path = "/media/data2/FakeAVCeleb_v1.2/" + str(row.get("Unnamed: 9", "")) + "/" + str(row["path"])
    destination = f"/media/data2/FakeAVCeleb_v1.2/FakeAVCeleb/val/real/{row['type']}-{row['source']}-{row['path']}"
    print("Attempting to copy from:", path)
    print("Destination path:", destination)
    if not os.path.isfile(path):
        print("File does not exist:", path)
    else:
        try:
            shutil.copy(path, destination)
        except Exception as e:
            print(f"Error copying file: {path}. Error: {e}")

for index, row in real_methods[350:].iterrows():
    path = "/media/data2/FakeAVCeleb_v1.2/" + str(row.get("Unnamed: 9", "")) + "/" + str(row["path"])
    destination = f"/media/data2/FakeAVCeleb_v1.2/FakeAVCeleb/train/real/{row['type']}-{row['source']}-{row['path']}"
    try:
        shutil.copy(path, destination)
    except FileNotFoundError:
        print(f"File not found: {path}, skipping...")

for index, row in shuffled_no_real_methods[:3000].iterrows():
    path = "/media/data2/FakeAVCeleb_v1.2/" + str(row.get("Unnamed: 9", "")) + "/" + str(row["path"])
    destination = f"/media/data2/FakeAVCeleb_v1.2/FakeAVCeleb/val/fake/{row['type']}-{row['source']}-{row['path']}"
    try:
        shutil.copy(path, destination)
    except FileNotFoundError:
        print(f"File not found: {path}, skipping...")

for index, row in shuffled_no_real_methods[3000:].iterrows():
    path = "/media/data2/FakeAVCeleb_v1.2/" + str(row.get("Unnamed: 9", "")) + "/" + str(row["path"])
    destination = f"/media/data2/FakeAVCeleb_v1.2/FakeAVCeleb/train/fake/{row['type']}-{row['source']}-{row['path']}"
    try:
        shutil.copy(path, destination)
    except FileNotFoundError:
        print(f"File not found: {path}, skipping...")
