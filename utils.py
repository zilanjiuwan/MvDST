import torch
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sparse
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from kmeans_gpu import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import scipy.io as sio
import scanpy as sc
import mnmstpy as mnmst
import os
import pandas as pd
import logging
import anndata
from scipy.sparse import csr_matrix
def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())
def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs

def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1
def load_graph_data(dataset_name, show_details=True):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    # section_id = "151507"
    # if show_details:
    #     print("++++++++++++++++++++++++++++++")
    #     print("---details of graph dataset---")
    #     print("++++++++++++++++++++++++++++++")
    # input_dir = os.path.join('F:/王海月/代码/UGIMC/Data/', section_id)
    # adata = sc.read_visium(path=input_dir, count_file=section_id + '_filtered_feature_bc_matrix.h5')

    section_id = 'V1'
    input_dir = os.path.join('F:\\王海月\\代码\\UGIMC\\Data\\Breast_Cancer', section_id)
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[:, adata.var['total_counts'] > 100]
    # sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    hvg_filter = adata.var['highly_variable']
    sc.pp.normalize_total(adata, inplace=True)
    adata = adata[:, hvg_filter]
    enhanced_adata, cell_spatial = mnmst.data_enhance(adata, k_nei=10, ratio=0.4)
    # dataset = "./dataset/" + "151507"
    dataset = "./dataset/" + "Breast"
    data = sio.loadmat('{}.mat'.format(dataset))
    feat = np.array(enhanced_adata.X)
    Y_list = []
    Y_list.append(np.squeeze(data['gt']))
    label = Y_list[0]
    adj = cell_spatial.A
    print(adj.shape)
    print(feat.shape)
    print("dataset name:   ", dataset_name)
    print("feature shape:  ", feat.shape)
    print("label shape:    ", label.shape)
    print("adj shape:      ", adj.shape)
    print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
    print("category num:          ", max(label)-min(label)+1)
    print("category distribution: ")
    for i in range(max(label).astype(int) +1):
        print("label", i, end=":")
        print(len(label[np.where(label == i)]))
    print("++++++++++++++++++++++++++++++")

    return feat, label, adj
def load_graph_datas(dataset_name, show_details=True):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    # section_id = "151670"
    # input_dir = os.path.join('F:/王海月/代码/UGIMC/Data/', section_id)
    # adata = sc.read_visium(path=input_dir, count_file=section_id + '_filtered_feature_bc_matrix.h5')
    section_id = 'V1'
    input_dir = os.path.join('F:\\王海月\\代码\\UGIMC\\Data\\Breast_Cancer', section_id)
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[:, adata.var['total_counts'] > 100]
    # sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    hvg_filter = adata.var['highly_variable']
    sc.pp.normalize_total(adata, inplace=True)
    adata = adata[:, hvg_filter]
    enhanced_adata, cell_spatial = mnmst.data_enhance(adata, k_nei=6, ratio=0.2)
    # dataset = "./dataset/" + '151670'
    dataset = "./dataset/" + 'Breast'
    data = sio.loadmat('{}.mat'.format(dataset))
    feat = np.array(enhanced_adata.X)
    Y_list = []
    Y_list.append(np.squeeze(data['gt']))
    label = Y_list[0]
    adj = cell_spatial.A
    adj_m = pd.read_csv('F:/王海月/代码/UGIMC/Denoising/output_histology_breast_cancer.csv') #output_histology_breast_cancer output_histology_151670
    adj_m = np.array(adj_m)
    print(adj_m.shape)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("adj_histopoloy shape: ",adj_m.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    return feat, label, adj, adj_m

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clustering(feature, true_labels, cluster_num):
    print(type(feature))
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cpu")
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    return round(100 * acc, 2), round(100 * nmi, 2), round(100 * ari, 2), round(100 * f1, 2), predict_labels.numpy()
