# Detection and Discovery of Misinformation Sources using Attributed Webgraphs  

### Interactive News Webgraph
To explore the webgraph related to these news sites, checkout our [interactive webgraph exploration tool](https://netneighbor.petercarragher.com/high-profile-news-network) built ontop of the CommonCrawl dataset.

### Introduction
These scripts can be used to train classifiers using the [NewsSEO dataset](https://kilthub.cmu.edu/articles/dataset/Dataset_for_Detection_and_Discovery_of_Misinformation_Sources_using_Attributed_Webgraphs_/25174193/1) and is based on the research paper "Detection and Discovery of Misinformation Sources using Attributed Webgraphs" [[PDF](https://arxiv.org/pdf/2401.02379.pdf)]. If you use, extend or build upon this project, please cite the following paper (upcoming at ICWSM 2024):
```
@article{carragher2024detection,
  title={Detection and Discovery of Misinformation Sources using Attributed Webgraphs},
  author={Carragher, Peter and Williams, Evan M and Carley, Kathleen M},
  journal={arXiv preprint arXiv:2401.02379},
  year={2024}
}
```

### Inputs
* Follow [the readme](data_collection/README.md) to populate the data directory with the [NewsSEO dataset](https://kilthub.cmu.edu/articles/dataset/Dataset_for_Detection_and_Discovery_of_Misinformation_Sources_using_Attributed_Webgraphs_/25174193/1)
* Webgraph data & SEO attributes have been pulled from ahrefs.com 
* Labels have been scraped from mediabiasfactcheck.com using this [open-source scraper](https://github.com/CASOS-IDeaS-CMU/media_bias_fact_check_data_collection)

### Environment Setup
```
pip3 install -r requirements.txt
# Generate edge weights
cd analysis && python3 weights.py && cd ../ 
# Run GNN weight scheme experiments
python3 gnns/train.py 0 
# Run GNN top N backlink experiments
python3 gnns/train.py 1
```

### Outputs
This code is provided as is, and neither the author nor the university is responsible for maintaining it. It provides the following functionality:
* A classifier that predicts the reliability of news sources
* A classifier that predicts the political leaning of news sources
* A discovery system that finds more unreliable news sources from an initial list of news sites

More specifically, the repository is organized as follows:
* analysis: 
    * [blogping.ipynb](analysis/blogping.ipynb): blogping / user generated features analysis
    * [country.ipynb](analysis/country.ipynb): country and continent breakdown of URLs based on hosting IP addresses
    * [link_scheme_identification.ipynb](analysis/link_scheme_identification.ipynb): algorithm 1 for the misinformation source discovery algorithm
    * [weights.py](analysis/weights.py): generate and save edge weight schemes to file
    * [weight_distributions.ipynb](analysis/weight_distributions.ipynb): plots for distribution of edge weights as computed in weights.ipynb
* data_collection: 
    * Ahrefs R API scripts: 
        * [ahref_backlinks.R](data_collection/ahref_backlinks.R): fetch backlinks for a given list of domains
        * [ahref_outlinks.R](data_collection/ahrefs_outlinks.R): fetch outlinks for a given list of domains
        * [ahref_nodes.R](data_collection/ahref_nodes.R): fetch attributes for a given list of domains
* evaluation:
    * [headlines.ipynb](evaluation/headlines.ipynb): a scraper built ontop of newspaper3k that outputs a .json file compatible with label_studio for manual evaluation of news articles
    * [krippendorf.ipynb](evaluation/krippendorf.ipynb): inter-annotator agreement between domain level reliability ratings
    * [label_studio_config.xml](evaluation/label_studio_config.xml): configuration for label studio that can create labeling jobs from json output of headlines.ipynb 
* flat_models:
    * [bias_removal.ipynb](flat_models/bias_removal.ipynb): analysis of bias removal techniques for political bias in dataset
    * [discovery.ipynb](flat_models/discovery.ipynb): runs the misinfo and bias classifiers on the link scheme outlinks.
        * First, outlinks must be generated from output of [link_scheme_identification.ipynb](analysis/link_scheme_identification.ipynb). 
        * Also trains the news source classifier. 
    * [train_classifiers.ipynb](flat_models/train_classifiers.ipynb): training & analysis of the misinfo & bias classifiers 
* gnns:
    * For weighted experiments, first run [weights.py](analysis/weights.py) for on both backlinks and outlinks
    * [model.py](gnns/model.py): defines the GCN used for predicting reliability & bias labels on the webgraphs
    * [seo_import.py](gnns/seo_import.py): sets up data imports, labels & batching for webgraphs
    * [train.py](gnns/train.py): trains the GCN defined in model.py using the data imported via [seo_import.py](gnns/seo_import.py) for all tasks, networks variants & edge weightings
    * [results.ipynb](gnns/results.ipynb): plots the results of the GNN experiments: heatmap & top N backlink plots (see results folder)
* politicalnews:
    * For evaluation, we also train our models on the [PoliticalNews dataset](https://osf.io/ez5q4/) as described in [Castelo et. al., 2019](https://arxiv.org/pdf/1905.00957.pdf) 
    * [discovery_seo.ipynb](politicalnews/discovery_seo.ipynb): implementation of the partial F1 evaluation metric as defined by [Chen & Freire, 2020](https://dl.acm.org/doi/abs/10.1145/3366424.3385772)
    * [train_misinfo.ipynb](politicalnews/train_misinfo.ipynb): comparison of misinfo classifiers trained with SEO features on the MBFC and politicalnews datasets respectively
* survival_rates:
    * url_list/: folder containing url lists used to train the parked domain classifiers
    * [parked_domain_sample.ipynb](survival_rates/parked_domain_sample.ipynb): scrapes positive examples of parked domains from [sedo.com](https://sedo.com/search/searchresult.php4?&language=us)
    * *_features.csv: features generated from the Parked Domain Classifier 
    * [parked_domain_classifier.ipynb](survival_rates/parked_domain_classifier.ipynb): trains the parked domain classifier with features from [Vissers et. al., 2015](https://github.com/flaiming/Domain-Parking-Sensors)
    * [requests.ipynb](survival_rates/requests.ipynb): sends GET requests to domain lists & checks responses

### License
BSD 3-Clause License

Copyright (c) 2024, Peter Carragher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
