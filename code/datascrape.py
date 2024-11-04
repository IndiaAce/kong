import os
import requests
import git
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

def setup_directories():
    # Create necessary directories for data storage
    directories = ["data/mitre_attack", "data/splunk_escu", "data/splunk_docs", "data/cleaned_data"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def clone_splunk_escu():
    # Clone the Splunk ESCU GitHub repository
    escu_repo_url = "https://github.com/splunk/security_content.git"
    escu_repo_path = "data/splunk_escu/security_content"
    
    if not os.path.exists(escu_repo_path):
        git.Repo.clone_from(escu_repo_url, escu_repo_path)
        print("Splunk ESCU repository cloned successfully.")
    else:
        print("Splunk ESCU repository already exists.")

def clone_mitre_attack():
    # Clone MITRE ATT&CK repository
    mitre_repo_url = "https://github.com/mitre/cti.git"
    mitre_repo_path = "data/mitre_attack/cti"
    
    if not os.path.exists(mitre_repo_path):
        git.Repo.clone_from(mitre_repo_url, mitre_repo_path)
        print("MITRE ATT&CK repository cloned successfully.")
    else:
        print("MITRE ATT&CK repository already exists.")

def scrape_splunk_docs():
    # Scrape specific Splunk Docs on SPL formatting, field names, and search commands
    base_urls = [
        "https://docs.splunk.com/Documentation/Splunk/latest/SearchReference/Searchcommands",
        "https://docs.splunk.com/Documentation/Splunk/latest/SearchReference/CommonEvalFunctions",
        "https://docs.splunk.com/Documentation/Splunk/latest/SearchReference/Fields",
        "https://docs.splunk.com/Documentation/Splunk/latest/SearchReference/UsingFields"
    ]
    
    splunk_docs_path = "data/splunk_docs"
    
    for base_url in base_urls:
        response = requests.get(base_url)
        if response.status_code == 200:
            page_name = base_url.split("/")[-1] + ".html"
            with open(os.path.join(splunk_docs_path, page_name), "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"Saved {page_name}")
        else:
            print(f"Failed to retrieve {base_url}. Status code: {response.status_code}")

def clean_html_content(file_path, output_path):
    # Read HTML content and remove unnecessary tags and formatting
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Remove script and style tags
    cleaned_content = re.sub(r"<script.*?>.*?</script>", "", content, flags=re.DOTALL)
    cleaned_content = re.sub(r"<style.*?>.*?</style>", "", cleaned_content, flags=re.DOTALL)
    
    # Remove HTML tags, keeping only the text
    cleaned_content = re.sub(r"<.*?>", "", cleaned_content)
    
    # Remove extra whitespace
    cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()
    
    # Write cleaned content to output file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned_content)
    print(f"Cleaned data saved to {output_path}")

def prepare_data_for_cleaning():
    # Prepare all scraped Splunk docs for data cleansing
    splunk_docs_path = "data/splunk_docs"
    cleaned_data_path = "data/cleaned_data"
    
    for file_name in os.listdir(splunk_docs_path):
        if file_name.endswith(".html"):
            file_path = os.path.join(splunk_docs_path, file_name)
            output_path = os.path.join(cleaned_data_path, file_name.replace(".html", "_cleaned.txt"))
            clean_html_content(file_path, output_path)

if __name__ == "__main__":
    setup_directories()
    clone_splunk_escu()
    clone_mitre_attack()
    scrape_splunk_docs()
    prepare_data_for_cleaning()
