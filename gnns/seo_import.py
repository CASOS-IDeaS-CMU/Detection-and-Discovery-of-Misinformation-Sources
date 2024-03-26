import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.loader import NeighborLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
import random
random.seed(123)
np.random.seed(123)
NO_LABEL = -9

def drop_corrupted_rows(links, labelled):
    """
    Drop rows with corrupted data- there were like 3 cryllic websites that the api couldn't ingest.
    They return nonsense- we'll delete them here.
    """
    a = links.domain_from.unique().tolist()
    b = links.domain_to.unique().tolist()
    uh = list(set(a +b))
    todrop_inlabeled = labelled[~labelled.url.isin(uh)].url
    labelled = labelled[~labelled.url.isin(todrop_inlabeled)]
    
    links = links[links.domain_from.isin(labelled.url.tolist())]
    links = links[links.domain_to.isin(labelled.url.tolist())]

    return links, labelled 

def import_seo_links(data_input_path, links_input_path, label_scheme, labels = None):
    """
    import data
    """
    attributes = pd.read_csv(data_input_path).drop_duplicates(subset='url')
    if not labels is None:
        if 'label' in attributes.columns:
            attributes.drop(columns='label', inplace=True)
        labelled = pd.merge(attributes, labels, how='left', on='url').fillna(NO_LABEL)
    else:
        assert('label' in attributes.columns)
        labelled = attributes

    links = pd.read_csv(links_input_path).dropna()
    links, labelled = drop_corrupted_rows(links, labelled)

    urls = labelled.url.unique().tolist()
    url_mapper = {url: i for i, url in enumerate(urls)}
    labelled['id'] = labelled['url'].map(url_mapper)
    links['source'] = links['domain_from'].map(url_mapper)
    links['target'] = links['domain_to'].map(url_mapper)

    links.drop(columns = ['domain_from', 'domain_to'], inplace=True)
    
    #lets try dropping 3 and 4's: 
    labelled.dropna(inplace=True)
    labelled['label'] = labelled.label.astype(int).replace(label_scheme)

    return labelled, links, url_mapper

def train_val_test_split(labelled, links, weight_scheme):
    """
    Generate train, val, and test masks and y tensor
    """
    targets = labelled[labelled['label'] > -1]
    ids = targets[['id', 'label']]
    X_train, X_v = train_test_split(ids, test_size=0.20, random_state=42, shuffle=True, stratify = ids.label)
    X_val, X_test = train_test_split(X_v, test_size=0.003, random_state=42, shuffle=True, stratify = X_v.label)

    train_mask = torch.tensor(X_train.id.values).long()
    val_mask = torch.tensor(X_val.id.values).long()
    test_mask = torch.tensor(X_test.id.values).long()

    y = torch.tensor(labelled.label.to_numpy()).long()

    # normalize feature matrix
    features = labelled.copy().drop(columns = ['url', 'id', 'label']).values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(np.log(features+1))
    x_scaled = torch.FloatTensor(x_scaled)
    
    link_attrs = links.loc[:, [x for x in list(set(['links', weight_scheme])) if not x in ['none','log_links']]]
    edge_list = torch.tensor([links['source'].values, links['target'].values]).long()
    if 'log' in weight_scheme or 'none' in weight_scheme:
        edge_weight = torch.FloatTensor(np.log(link_attrs['links'].values+1))
    else:
        edge_weight = torch.FloatTensor(link_attrs[weight_scheme].values)

    data = Data(x=x_scaled, edge_index = edge_list, edge_weight=edge_weight, y=y, train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)

    # sample neighbors - we use this for minibatching
    train_loader = NeighborLoader(data, input_nodes=(data.train_mask),
                                num_neighbors=[25, 15], batch_size=64, shuffle=True,
                                num_workers=0)
    valid_loader = NeighborLoader(data, input_nodes = (data.val_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(data, input_nodes = (data.test_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)

    return train_loader, valid_loader, test_loader


def data_only(labelled, links):
    """
    Generate train, val, and test masks and y tensor
    """
    targets = labelled[labelled['label'] > -1]
    ids = targets[['id', 'label']]
    X_train, X_v = train_test_split(ids, test_size=0.20, random_state=42, shuffle=True, stratify = ids.label)
    X_val, X_test = train_test_split(X_v, test_size=0.5, random_state=42, shuffle=True, stratify = X_v.label)

    train_mask = torch.tensor(X_train.id.values).long()
    val_mask = torch.tensor(X_val.id.values).long()
    test_mask = torch.tensor(X_test.id.values).long()

    y = torch.tensor(labelled.label.to_numpy()).long()

    # normalize feature matrix
    features = labelled.copy().drop(columns = ['url', 'id', 'label']).values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(features)
    x_scaled = torch.FloatTensor(x_scaled)

    edge_list = torch.tensor([links['source'].values, links['target'].values]).long()

    edge_weight = torch.FloatTensor(np.log([links['links'].values]))

    data = Data(x=x_scaled, edge_index = edge_list, edge_weight=edge_weight.reshape(-1), y=y, train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)
    return data
