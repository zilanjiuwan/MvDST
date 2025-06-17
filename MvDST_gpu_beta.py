import os
import torch
import time
import psutil
import numpy as np
import scipy.sparse as sp
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from model import my_model
from utils import preprocess_graph, setup_seed, clustering, get_logger

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

def MvDST(features, true_labels, adj,  device='cuda:0', gnnlayers=6, epochs=400, dims=[500], lr=1e-3, sigma=0.01, cluster_num=7):
    """
    Multi-view Denoising Spatial Transcriptomics (MvDST) model training.

    Args:
        features (np.ndarray): Feature matrix (n_samples x n_features)
        true_labels (np.ndarray): Ground truth labels
        adj (sp.csr_matrix): Adjacency matrix of primary view
        adj_m (sp.csr_matrix): Adjacency matrix of auxiliary view
        device (str): Device to run the model on ('cuda:0' or 'cpu')
        gnnlayers (int): Number of graph smoothing layers
        epochs (int): Number of training epochs
        dims (list): Hidden dimensions for the model
        lr (float): Learning rate
        sigma (float): Gaussian noise std
        cluster_num (int): Number of clusters

    Returns:
        decoder_output (np.ndarray): Reconstructed feature output
        best_pred_labels (np.ndarray): Predicted labels with best ARI
    """

    logger = get_logger()
    logger.info('--------------------MvDST Training Start--------------------')

    # Convert adjacency matrices to sparse CSR format if not already
    adj = sp.csr_matrix(adj)

    # Remove self-loops and eliminate zeros
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Laplacian smoothing
    logger.info('Laplacian smoothing...')
    adj_norm_s = preprocess_graph(adj, gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()

    # Perform smoothing over gnnlayers
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)

    sm_fea_s = torch.FloatTensor(sm_fea_s).to(device)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

    # Start training
    best_acc = best_nmi = best_ari = best_f1 = 0
    best_labels = None
    setup_seed(0)
    model = my_model([features.shape[1]] + dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='sum')

    inx = sm_fea_s
    target = torch.FloatTensor(adj_1st).to(device)

    logger.info('--------------------Start Training Loop--------------------')
    training_start = time.time()
    for epoch in tqdm(range(epochs)):
        model.train()
        z1, z2, decoder_out = model(inx, is_train=True, sigma=sigma)
        S = z1 @ z2.T
        loss = 10 * F.mse_loss(S, target) + criterion(decoder_out, inx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z1, z2, decoder_out = model(inx, is_train=False, sigma=sigma)
                hidden_emb = (z1 + z2) / 2
                acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, cluster_num)
                if ari > best_ari:
                    best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1
                    best_labels = predict_labels

    total_time = time.time() - training_start
    logger.info(f"Training Complete. Best ARI: {best_ari:.4f}, Time: {format_time(total_time)}")
    # Memory stats
    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    logger.info(f"Peak RAM usage: {mem_used:.3f} GB")
    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        logger.info(f"Peak GPU memory usage: {peak_gpu:.3f} GB")

    return decoder_out.detach().cpu().numpy(), best_labels, hidden_emb.detach().cpu().numpy()
