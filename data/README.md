----------------------------------------
GENERAL INFORMATION
----------------------------------------

1. Title of Dataset: NewsSEO Dataset

2. Author Information

Author Contact Information
    Name: Peter Carragher
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: petercarragher@cmu.edu


Author Contact Information
    Name: Evan M. Williams
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: emwilla@andrew.cmu.edu

Author Contact Information 
    Name: Kathleen M. Carley
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: carley@andrew.cmu.edu

---------------------------------------
DATA & FILE OVERVIEW
---------------------------------------

Directory of Files:

   A. Filename:  filtered_attrs.csv
   
      Short description:  This CSV file contains reliability labels and SEO attributes for each of the 3211 news domains in the study. Reliability labels are from 1 (low) to 6 (high) following the mediabiasfactcheck (MBFC) [reliability scale](https://mediabiasfactcheck.com/methodology/). SEO attributes are collected via a call to the [Ahrefs metrics-extended API](https://ahrefs.com/api/documentation/metrics-extended).


   B. Filename:  bias_labels.csv
   
      Short description:  This CSV file contains bias labels for 2629 of the news domains that have [MBFC bias scores](https://mediabiasfactcheck.com/methodology/), ranging from extreme left (-2) to extreme right (+2). 


   C. Filename:  filtered_backlinks.csv   
   
      Short description:  This CSV file contains the top 10 backlinking domains for each of the 3211 news domains, collected via a call to the [Ahrefs referring domains API](https://ahrefs.com/api/documentation/refdomains).

        
   D. Filename:  filtered_outlinks.csv  
   
      Short description: This CSV file contains the top 10 outlinking domains for each of the 3211 news domains, collected via a call to the [Ahrefs linked domains API](https://ahrefs.com/api/documentation/linked-domains).


   E. Filename:  filtered_combined_attrs.csv
   
      Short description: This CSV file contains SEO attributes for the 3211 news domains, as well as their backlinking (filtered_backlinks.csv) and outlinking domains (filtered_outlinks.csv), collected via a call to the [Ahrefs metrics-extended API](https://ahrefs.com/api/documentation/metrics-extended).


   F. Filename:  link_scheme_outlinks.csv   
   
      Short description: This CSV file contains the top 100 outlinking domains for each of link schemes identified within the backlinking domains in filtered_backlinks.csv, collected via a call to the [Ahrefs linked domains API](https://ahrefs.com/api/documentation/linked-domains). Algorithm 1 in the paper details the indentification of these link schemes.

        
   G. Filename:  link_scheme_outlink_attrs.csv  
   
      Short description: This CSV file contains SEO attributes for each of the link scheme outlinks in link_scheme_outlinks.csv, collected via a call to the [Ahrefs metrics-extended API](https://ahrefs.com/api/documentation/metrics-extended).


   H. Filename:  discovered_domains.csv  

      Short description: This CSV file contains the subset of domains that were classified as unreliable from the link scheme outlinks (link_scheme_outlink_attrs.csv).


   I. Filename:  discovered_domains_sample_annotated.csv  

      Short description: This CSV file contains a sample of the discovered domains (discovered_domains.csv) that are annotated by the authors for political bias and reliability.



These data are provided courtesy of Ahrefs.com.

File Naming Convention: 
* *_attrs.csv: attributed node list
* *links.csv: <source, destination> link list

-------------------------------------------------------
METHODOLOGICAL INFORMATION
-------------------------------------------------------

1. Software-specific information:

Name: R (https://www.r-project.org/)
Version: 4.3.2
System Requirements: N/A
Open Source? (Y/N):  Y

Additional Notes: Data were pulled from the Ahrefs API using the [RAhrefs package](https://github.com/Leszek-Sieminski/RAhrefs).

Name: [MBFC Scraper](https://github.com/CASOS-IDeaS-CMU/media_bias_fact_check_data_collection)
Version: 1
System Requirements: N/A
Open Source? (Y/N):  Y

2. Equipment-specific information:

Manufacturer: Dell 
Model: Precision Tower 3660

3. Date of data collection: 20230601 - 20230701

--------------------------------------------------
NOTES ON REPRODUCIBILITY 
--------------------------------------------------

Webgraphs are dynamic, and so attempts to reproduce this dataset will have more up-to-date attributes, backlinks, and outlinks, reflecting changes to the structure of the news domains since the time of this study.
Relevant scripts in the GitHub repository will allow this research to be reproduced with values dependent upon these changes:
* data/ahref_backlinks.R: fetch backlinks for a given list of domains
* data/ahref_outlinks.R: fetch outlinks for a given list of domains
* data/ahref_nodes.R: fetch attributes for a given list of domains
