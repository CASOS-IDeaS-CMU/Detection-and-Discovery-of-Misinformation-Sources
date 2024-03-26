import sys
from os.path import dirname
sys.path.append(dirname(__file__))
sys.path.append(".")

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from gnns.seo_import import *
from sklearn.metrics import  accuracy_score, f1_score
from torch_geometric.seed import seed_everything
from gnns.model import GNN_v2
import random
import sys

EPOCHS = 1000
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_params():
    with torch.no_grad():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device)
        initial_x = batch.x.clone()
        model(batch.x, batch.edge_index, batch.edge_weight, initial_x)


def train():
    """
    Trains the model
    """
    model.train()
    total_loss = total_correct = 0
    train_loader_len = 0
    possible_correct = 0
    predictions = []
    true_labs = []
    
    for batch in tqdm(train_loader):

        batch_size = batch.batch_size
        optimizer.zero_grad()
        batch = batch.to(device)
        initial_x = batch.x.clone()
        out = model(batch.x, batch.edge_index, batch.edge_weight, initial_x)[:batch_size]
        loss = loss_fn(out, batch.y[:batch_size])
        
        loss.backward()
        optimizer.step()
        
        pred = out.argmax(dim=-1)
        
        possible_correct += batch_size
        total_correct += int((pred == batch.y[:batch_size]).sum())
        
        total_loss += float(loss)
        train_loader_len += 1
        
        predictions.append(pred.cpu().numpy())
        true_labs.append(batch.y[:batch_size].cpu().numpy())
        

    train_loss = total_loss / train_loader_len
    train_acc = accuracy_score(np.concatenate(true_labs), np.concatenate(predictions))
    f1 = f1_score(np.concatenate(true_labs), np.concatenate(predictions), average = 'macro')

    return train_loss, train_acc, f1

def test(loader):
    with torch.no_grad():
        model.eval()
        true_labs = []
        total_loss = 0
        train_loader_len = 0
        predictions = []
        true_labs = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            batch_size = batch.batch_size
            initial_x = batch.x.clone()
            out = model(batch.x, batch.edge_index, batch.edge_weight, initial_x)[:batch_size]
            pred = out.argmax(dim=-1)

            # calculate validation loss
            loss = loss_fn(out, batch.y[:batch_size])
            pred = out.argmax(dim=-1)
                        
            total_loss += float(loss)
            train_loader_len += 1
            
            predictions.append(pred.cpu().numpy())
            true_labs.append(batch.y[:batch_size].cpu().numpy())
        
    valid_loss = total_loss / train_loader_len
    valid_acc = accuracy_score(np.concatenate(true_labs), np.concatenate(predictions))
    f1 = f1_score(np.concatenate(true_labs), np.concatenate(predictions), average = 'macro')

    return valid_loss, valid_acc, f1


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_epoch = 0

    def __call__(self, train_loss, validation_loss):
        
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:# and self.counter > 40:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def prepare_top_n_networks(df):
    def get_top_n_rows(df, i):
        rows_to_extract = [r for r in range(len(df)) if r % 10 < i]
        return df.iloc[rows_to_extract]    
    
    for i in range(1, 11, 1):
        get_top_n_rows(df, i).to_csv(f"./data/top_{i}.csv")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <experiment_flag>")
        sys.exit(1)
    arg = sys.argv[1]
    if arg not in ('0', '1'):
        print("Invalid boolean value. Please provide either '0' or '1'.")
        sys.exit(1)
    run_top_n_experiment = bool(int(arg))
    print(run_top_n_experiment)
    if run_top_n_experiment:
        output_prefix = 'results/top_n_experiment'
        weight_schemes = ['none', 'links', 'log_links']
        lr = 0.01
        prepare_top_n_networks(pd.read_csv('./data/filtered_backlinks.csv'))
        link_networks = {
            f"top_{i}": ('./data/filtered_combined_attrs.csv', f"./data/top_{i}.csv") for i in range(1,11,1)
        }
    else:
        output_prefix = 'results/gnn_weight_experiment'
        weight_schemes = ['none', 'links', 'log_links', 'unique_pages','tb_ratio','e_tb_ratio','so_ratio', 'e_so_ratio','tp_ratio','sp_ratio']
        lr = 0.05
        link_networks = {
            'backlinks':('./data/filtered_combined_attrs.csv', './analysis/weight_backlinks.csv'),
            'outlinks': ('./data/filtered_combined_attrs.csv', './analysis/weight_outlinks.csv'),
            'combined': ('./data/filtered_combined_attrs.csv', './analysis/weight_combined.csv'),
        }

    reliability_labels = './data/filtered_attrs.csv'
    bias_labels = './data/bias_labels.csv'
    label_schemes = {
        'reliability': (reliability_labels, {6:1,5:1,4:0,3:0,1:0, NO_LABEL:-1}), # reliability labels
        'abs_bias': (bias_labels, {-2:1,-1:0,0:0,1:0,2:1, NO_LABEL:-1}), # absolute bias: extreme vs centrist
        'rel_bias': (bias_labels, {-2:0,-1:0,0:-1,1:1,2:1, NO_LABEL:-1}), # relative bias: left vs right
    } 

    training_stats = {}
    final_pds = []
    for (network, (data_input_path, links_input_path)) in link_networks.items():
        for (task, (labels_input_path, label_scheme)) in label_schemes.items():
            for weight_scheme in weight_schemes:
                training_stats[weight_scheme] = []

                labels = pd.read_csv(labels_input_path)
                if 'bias' in labels.columns:
                    labels = labels[['url', 'bias']]
                else:
                    labels = labels[['url', 'label']]
                labels.columns=['url','label']
                labelled, links, url_mapper = import_seo_links(data_input_path, links_input_path, label_scheme, labels)

                if weight_scheme == 'so_ratio':
                    links[['so_ratio']] = links[['so_ratio']].replace(np.inf, 1)

                train_loader, valid_loader, test_loader = train_val_test_split(labelled, links, weight_scheme)
                model = GNN_v2(23, 2, use_weights=(weight_scheme != 'none'))
                model = model.to(device)
                init_params()
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
                loss_fn = torch.nn.CrossEntropyLoss()

                early_stopping = EarlyStopping(patience=30, min_delta=0.0001)
                for epoch in range(0, EPOCHS):

                    tloss, tacc, tf1 = train()
                    vloss, vacc, vf1 = test(valid_loader)
                    _, test_accuracy, test_f1 = test(test_loader)
                    training_stats[weight_scheme].append(
                        {
                            'epoch': epoch + 1,
                            'Training Accuracy': tacc,
                            'Training Loss': tloss,
                            'Training F1': tf1,
                            'Valid. Loss': vloss,
                            'Valid. Accur.': vacc,
                            'Valid. F1': vf1,
                            'Test Accuracy' : test_accuracy,
                            'Test F1': test_f1
                        }
                    )
                    
                    early_stopping(tloss, vloss)
                    if early_stopping.early_stop:
                        print("We are at epoch:", epoch)
                        patience = early_stopping.patience
                        acc_csv = {
                            'network': network,
                            'task': task,
                            'weight_scheme': weight_scheme,
                            'validation_accuracy' : training_stats[weight_scheme][epoch-patience]['Valid. Accur.'],
                            'val_f1':training_stats[weight_scheme][epoch-patience]['Valid. F1'],
                            'training_accuracy': training_stats[weight_scheme][epoch-patience]['Training Accuracy'],
                            'train_f1': training_stats[weight_scheme][epoch-patience]['Training F1'], 
                            'test_accuracy': training_stats[weight_scheme][epoch-patience]['Test Accuracy'],
                            'test_f1':training_stats[weight_scheme][epoch-patience]['Test F1'],
                            'epoch': epoch - patience,
                        }
                        break
                
                if not early_stopping.early_stop:
                    acc_csv = {
                        'network': network,
                        'task': task,
                        'weight_scheme': weight_scheme,
                        'validation_accuracy' : training_stats[weight_scheme][EPOCHS-1]['Valid. Accur.'],
                        'val_f1':training_stats[weight_scheme][EPOCHS-1]['Valid. F1'],  
                        'training_accuracy': training_stats[weight_scheme][EPOCHS-1]['Training Accuracy'],
                        'train_f1': training_stats[weight_scheme][EPOCHS-1]['Training F1'], 
                        'test_accuracy': training_stats[weight_scheme][EPOCHS-1]['Test Accuracy'],
                        'test_f1':training_stats[weight_scheme][EPOCHS-1]['Test F1'],
                        'epoch': EPOCHS,
                    }
                final_pds.append(acc_csv)
                pd.DataFrame(final_pds).to_csv(output_prefix + '_results.csv')
                
    pd.DataFrame(final_pds).to_csv(output_prefix + '_results.csv', index=False)