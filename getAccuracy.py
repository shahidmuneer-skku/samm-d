import pandas as pd
import torch
def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:
                file_path = parts[0]
                try:
                    accuracy = float(parts[1])
                    data.append((file_path, accuracy))
                except ValueError:
                    print(f"Invalid accuracy value on line: {line.strip()}")
    return data

# Usage
file_name = '/home/shahid/LAV-DF-Shahid/scores2/scores_mfcc_randomseed_42.txt'
acc_data = read_file(file_name)
data = pd.read_json("/home/shahid/dfdc_preview_set/dataset.json")
correct_predictions = 0
total_samples = 0
for acc in acc_data:
    path = acc[0]
    start_index = path.find("dfdc_preview_set")
    relative_path = path[start_index+17:]
    label = data[relative_path]["label"]
    label = 0 if label=="real" else 1
    pred = acc[1]

    pred = torch.sigmoid(pred)
    print(pred)
    pred_class = 1 if pred > -2 else 0  
    prediction = 1 if pred_class == label else 0 
    correct_predictions += prediction
    total_samples += 1


accuracy = correct_predictions / total_samples
print(f"Total Accuracy is {accuracy*100:2f}")
