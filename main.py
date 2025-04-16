import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model
import torch.nn.functional as F
import pandas as pd
from utils import get_logger
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=6, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
parser.add_argument('--dataset', type=str, default='scData', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=20, help='type of dataset.')
parser.add_argument('--device', type=str, default='cpu', help='device')

args = parser.parse_args()

for args.dataset in ['scData']:
    print("Using {} dataset".format(args.dataset))
    file = open("result_baseline.csv", "a+")
    print(args.dataset, file=file)
    file.close()
    logger = get_logger()
    logger.info('Dataset:' + str('Breast'))
    # load data
    X, y, A, A_m = load_graph_datas(args.dataset, show_details=False)
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)
    adj_m = sp.csr_matrix(A_m)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_m = adj_m - sp.dia_matrix((adj_m.diagonal()[np.newaxis, :], [0]), shape=adj_m.shape)
    adj_m.eliminate_zeros()
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    if os.path.exists(path):
        sm_fea_s = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
    adj_m_1st = (adj_m + sp.eye(adj_m.shape[0])).toarray()

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    best_acc = 0
    best_nmi = 0
    best_ari = 0
    best_f1 = 0

    for seed in range(3):
        setup_seed(seed)
        best_acc1, best_nmi1, best_ari1, best_f11, prediect_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
        print(best_ari)
        model = my_model([features.shape[1]] + args.dims)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(args.device)
        inx = sm_fea_s.to(args.device)
        target = torch.FloatTensor(adj_1st).to(args.device)
        target_m = torch.FloatTensor(adj_m_1st).to(args.device)
        criterion = torch.nn.MSELoss(reduction='sum')
        print('Start Training...')
        logger.info('--------------------Start Training--------------------')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            z1, z2,decoder_out= model(inx, is_train=True, sigma=args.sigma)

            S = z1 @ z2.T
            loss =0.0001* (0.5 * F.mse_loss(S, target) + 0.5 * F.mse_loss(S, target_m) )+ criterion(decoder_out, inx) #+ criterion(decoder_out2, inx)
            # print(loss)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                z1, z2,decoder_out = model(inx, is_train=False, sigma=args.sigma)
                hidden_emb = (z1 + z2) / 2
                acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                if ari >= best_ari:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
                    best_label = predict_labels
                    best_h = hidden_emb.detach().numpy()
                    df_S = pd.DataFrame(best_h)
                    df_l = pd.DataFrame(best_label)
                    print(df_S.shape)
                    print(decoder_out.detach().numpy().shape)
                    df_X = pd.DataFrame(decoder_out.detach().numpy())
        tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        file = open("result_baseline.csv", "a+")
        logger.info('--------------------Training over--------------------')
        logger.info(""" ACC {:.2f} NMI {:.2f} ARI {:.2f} f_score {:.2f} """.format(best_acc,best_nmi,best_ari,best_f1))
        print(best_acc, best_nmi, best_ari, best_f1, file=file)
        file.close()
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)
    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)
    file = open("result_baseline.csv", "a+")
    print(args.gnnlayers, args.lr, args.dims, args.sigma, file=file)
    print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
    print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
    print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
    print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
    file.close()
