import os
import json
import csv
import yaml
import re


def setup_directories():
    # Create necessary directories for cleaned and processed data storage
    directories = ["data/processed_data"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def parse_mitre_attack():
    # Parse and extract relevant information from the MITRE ATT&CK corpus
    mitre_data_path = "data/mitre_attack/cti/enterprise-attack/attack-pattern"
    output_file = "data/processed_data/mitre_attack_cleaned.json"
    mitre_data = []

    for root, _, files in os.walk(mitre_data_path):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for obj in data.get("objects", []):
                        if obj.get("type") == "attack-pattern":
                            mitre_data.append({
                                "name": obj.get("name"),
                                "description": obj.get("description"),
                                "kill_chain_phases": obj.get("kill_chain_phases", []),
                                "external_references": obj.get("external_references", []),
                            })

    # Save extracted data to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mitre_data, f, indent=4)
    print(f"MITRE ATT&CK data saved to {output_file}")


def clean_splunk_escu():
    # Clean and structure detection rules from the Splunk ESCU repository
    escu_data_path = "data/splunk_escu/security_content/detections"
    output_file = "data/processed_data/splunk_escu_cleaned.json"
    escu_data = []

    for root, _, files in os.walk(escu_data_path):
        for file in files:
            if file.endswith(".yml"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    try:
                        data = yaml.safe_load(f)
                        escu_data.append({
                            "name": data.get("name"),
                            "description": data.get("description"),
                            "search": data.get("search"),
                            "tags": data.get("tags", {})
                        })
                    except yaml.YAMLError as e:
                        print(f"Error parsing {file}: {e}")

    # Save extracted data to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(escu_data, f, indent=4)
    print(f"Splunk ESCU data saved to {output_file}")


def clean_splunk_docs():
    # Extract useful content from the Splunk SPL documentation
    splunk_docs_path = "data/cleaned_data"
    output_file = "data/processed_data/splunk_docs_cleaned.csv"
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["document", "content"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_name in os.listdir(splunk_docs_path):
            if file_name.endswith("_cleaned.txt"):
                file_path = os.path.join(splunk_docs_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    writer.writerow({"document": file_name.replace("_cleaned.txt", ""), "content": content})
    print(f"Splunk SPL documentation data saved to {output_file}")


if __name__ == "__main__":
    setup_directories()
    parse_mitre_attack()
    clean_splunk_escu()
    clean_splunk_docs()
