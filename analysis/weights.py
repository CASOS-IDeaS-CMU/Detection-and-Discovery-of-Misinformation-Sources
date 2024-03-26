import networkx as nx
import pandas as pd
from functools import reduce
import numpy as np
from enum import Enum
from sklearn import neighbors

def generate_edge_weights(output_file, edge_df):
    edge_df = edge_df.groupby(['domain_from', 'domain_to'])
    edge_df = edge_df.max()
    edge_df.reset_index(inplace=True)
    edge_df['domain_from'] = edge_df['domain_from'].str.lower()
    edge_df['domain_to'] = edge_df['domain_to'].str.lower()

    node_mapping = {k: v for v, k in enumerate(set(list(edge_df.domain_from.unique()) + list(edge_df.domain_to.unique())))}

    edge_df['domain_from_idx'] = edge_df.domain_from.map(node_mapping)
    edge_df['domain_to_idx'] = edge_df.domain_to.map(node_mapping)

    G = nx.from_pandas_edgelist(edge_df, source='domain_from_idx', target='domain_to_idx', edge_attr=['links', 'unique_pages'], create_using=nx.DiGraph())

    labels = pd.read_csv('../data/filtered_combined_attrs.csv')
    labels['url'] = labels['url'].str.lower()
    inv_node_mapping = {v: k for k, v in node_mapping.items()}

    fails = []
    targets = []
    sources = []
    for (source, target) in G.edges:
        try:
            links = G[source][target]['links']
            
            target_backlinks = labels.loc[labels['url'] == inv_node_mapping[target]]['backlinks'].values[0]
            target_backlink_ratio = links / target_backlinks
            G[source][target]['tb_ratio'] = target_backlink_ratio

            source_outlinks = labels.loc[labels['url'] == inv_node_mapping[source]]['links_external'].values[0]
            source_outlink_ratio = links / source_outlinks
            if source_outlink_ratio == np.inf:
                source_outlink_ratio = 1
            G[source][target]['so_ratio'] = source_outlink_ratio

            pages = G[source][target]['unique_pages']
            target_ref_domains = labels.loc[labels['url'] == inv_node_mapping[target]]['refpages'].values[0]
            refpage_ratio = pages / target_ref_domains
            G[source][target]['tp_ratio'] = refpage_ratio

            source_ref_domains = labels.loc[labels['url'] == inv_node_mapping[source]]['pages'].values[0]
            source_refpage_ratio = pages / source_ref_domains
            G[source][target]['sp_ratio'] = source_refpage_ratio

            targets.append(target)
            sources.append(source)

        except Exception as e:
            fails.append((source, target))
            print(source, target, G[source][target]['links'], G[source][target]['unique_pages'])

    for target in targets:
        preds = list(G.predecessors(target))
        tb_ratio_sum = sum([G[pred][target]['tb_ratio'] for pred in preds])
        for pred in preds:
            G[pred][target]['e_tb_ratio'] = G[pred][target]['tb_ratio'] / tb_ratio_sum
    
    for source in sources:
        succs = list(G.neighbors(source))
        try:
            so_ratio_sum = sum([G[source][succ]['so_ratio'] for succ in succs])
            for succ in succs:
                G[source][succ]['e_so_ratio'] = G[source][succ]['so_ratio'] / so_ratio_sum
        except Exception as e:
            for succ in succs:
                G[source][succ]['so_ratio'] = 0
                G[source][succ]['e_so_ratio'] = 0

    edges = []
    for (source, target) in G.edges:
        attrs = G[source][target]
        try:
            edges.append([inv_node_mapping[source], inv_node_mapping[target], attrs['links'], attrs['unique_pages'], attrs['tb_ratio'], attrs['so_ratio'], attrs['tp_ratio'], attrs['sp_ratio'], attrs['e_tb_ratio'], attrs['e_so_ratio']])
        except Exception as e:
            print(inv_node_mapping[source], inv_node_mapping[target])
    fields = ['domain_from','domain_to','links','unique_pages','tb_ratio', 'so_ratio', 'tp_ratio', 'sp_ratio', 'e_tb_ratio', 'e_so_ratio']

    import csv
    with open(output_file, 'w') as f:
        write = csv.writer(f)    
        write.writerow(fields)
        write.writerows(edges)

if __name__ == '__main__':
    print('Generating Outlink Weights')
    generate_edge_weights('weight_outlinks.csv', pd.read_csv('../data/filtered_outlinks.csv'))
    print('Generating Backlink Weights')
    generate_edge_weights('weight_backlinks.csv', pd.read_csv('../data/filtered_backlinks.csv'))

    outlinks = pd.read_csv('weight_outlinks.csv')
    backlinks = pd.read_csv('weight_backlinks.csv')
    pd.concat([outlinks, backlinks], axis=0).to_csv('weight_combined.csv')
