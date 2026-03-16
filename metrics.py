# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 09:09:56 2026

@author: granged
"""

import numpy as np
import utils
from tqdm import tqdm
from sklearn.covariance import graphical_lasso
import requests
import networkx as nx


def outer_product_dist(data):
    patient_matrices = [np.outer(vector, vector)  for vector in data]
    
    n_patients = data.shape[0]

    distance_matrix = np.zeros((n_patients, n_patients))

    for i in tqdm(range(n_patients)):
        for j in range(i):
            distance_matrix[i,j] = utils.BW(patient_matrices[i], patient_matrices[j])
            
    distance_matrix += distance_matrix.T
    
    return distance_matrix



def graphical_lasso_dist(data):
    emp_cov = np.cov(data.T)
    emp_cov, _ = graphical_lasso(emp_cov, alpha=0.03)
    adj = emp_cov > 0

    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    n_patients = data.shape[0]

    patient_matrices = [np.outer(vector, vector) * adj  for vector in data]


    for i in tqdm(range(n_patients)):
        for j in range(i):
            distance_matrix[i,j] = utils.BW(patient_matrices[i], patient_matrices[j])
            
    distance_matrix += distance_matrix.T
    
    return distance_matrix

def string_db_dist(df):
    genes = df.columns.to_list()
    
    n_features = len(genes)
    n_patients = df.shape[0]

    string_api_url = "https://version-12-0.string-db.org/api"
    output_format = "json"
    method = "network"

    params = {
        "identifiers" : "\r".join(genes), # your protein list
        "species" : 9606 # NCBI/STRING taxon identifier 
    }

    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)

    adj = np.zeros((n_features, n_features))

    # Create graph
    G = nx.Graph()
    for edge in results.json():
        try:
            a = edge["preferredName_A"]
            a_idx = genes.index(str(a))
            b = edge["preferredName_B"]
            b_idx = genes.index(str(b))
            adj[a_idx,b_idx] = 1
            adj[b_idx,a_idx] = 1
            score = edge["score"]
            G.add_edge(a, b, weight=score)
        except:
            print(f'Problem with edge {edge}')

    distance_matrix = np.zeros((df.shape[0], df.shape[0]))
    n_patients = df.shape[0]

    patient_matrices = [np.outer(vector, vector) * adj  for vector in df.to_numpy()]


    for i in tqdm(range(n_patients)):
        for j in range(i):
            distance_matrix[i,j] = utils.BW(patient_matrices[i], patient_matrices[j])
            
    distance_matrix += distance_matrix.T

    return distance_matrix