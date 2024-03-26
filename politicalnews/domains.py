# This script converts the PoliticalNews dataset into the format found in data/domains.csv (https://osf.io/ez5q4/)

import os
import pandas as pd
from urllib.parse import urlparse

def extract_domains_from_urls(urls):
    domains = set()
    for url in urls:
        parsed_url = urlparse(url)
        domains.add(parsed_url.netloc)
    return domains

def load_csv_files(directory):
    all_domains = set()
    
    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    for file in csv_files:
        file_path = os.path.join(directory, file)
        
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Extract domains from the first column
        urls = df.iloc[:, 0].tolist()
        domains = extract_domains_from_urls(urls)
        
        # Add the unique domains to the set
        all_domains.update(domains)
    
    return list(all_domains)

# Directory containing the CSV files
directory = 'data/msm/'

# Load CSV files and extract unique domains
unique_domains = load_csv_files(directory)

import csv

def write_list_to_csv(data_list, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for data in data_list:
            writer.writerow([data, 0])

csv_file_path = 'domains_msm.csv'

# Write the list to the CSV file
write_list_to_csv(unique_domains, csv_file_path)
