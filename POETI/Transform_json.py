import json

# Load the JSON file
file_path = "/mnt/data/Vasile Alecsandri - Poezii.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Pretty format the JSON data
formatted_json_path = "/mnt/data/Vasile_Alecsandri_Poezii_Formatted.json"
with open(formatted_json_path, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

# Provide the formatted file to the user
formatted_json_path
