import copy
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import scipy
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from plugins import *
import warnings
import torch.nn.functional as F
import GPUtil
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from typing import Union, Tuple, Dict, List, Optional
import torch.nn as nn
import pandas as pd
import math
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import gc
from typing import Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, \
    fowlkes_mallows_score, accuracy_score, silhouette_score, adjusted_mutual_info_score
from sklearn.metrics import f1_score, precision_score, balanced_accuracy_score,recall_score,matthews_corrcoef
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from plugins import *
from scipy.stats import rankdata
import datetime


def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    
    try:
        torch.manual_seed(seed)  
        torch.cuda.manual_seed(seed) 
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  
        
    except ImportError:
        raise UserWarning('failed to set pytorch seed')

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_available_gpus(min_memory_mb=1000):
    all_gpus = GPUtil.getGPUs()
    available_gpus = [gpu.id for gpu in all_gpus if gpu.memoryFree >= min_memory_mb]
    available_gpus_objs = [gpu for gpu in all_gpus if gpu.memoryFree >= min_memory_mb]
    available_gpus_objs.sort(key=lambda x: x.memoryFree, reverse=True)
    available_gpus = [gpu.id for gpu in available_gpus_objs]

    return available_gpus


def flatten_ndiags(A,n=8,rm_main=False):
    """
    args:
        A: upper triangle formation of all cell's contact matrix, shape:(cell_num,chrsize,chrsize) eg.(626,250,250)
    func:
        get n diags from the upper triangle formation of the contact matrix and flatten them to a vector
        get all diags when n=-1
    """
    if isinstance(rm_main, bool):
        rm_main = 1 if rm_main else 0
    if A.ndim == 2:
        A = A[None,:]
        reshape=True
    else:
        reshape=False

    if 0 < n < 1:
        n = int(n*A.shape[-1])
        n = min(max(n, 5), 15)
        print(n)
    if n == -1 or n > A.shape[-1]:
        n = A.shape[-1]
    n = min(n, A.shape[-1])
    upper_diags = []
    for i in range(rm_main, n):
        upper_diag = np.diagonal(A, offset=i, axis1=-2, axis2=-1)
        upper_diags.append(upper_diag)

    B = np.concatenate(upper_diags,axis=1)

    if reshape:
        B = B[0]

    return B


def unflatten_all_ndiags(B: np.ndarray, chrsize: int) -> np.ndarray:
    cell_num, total_length = B.shape
    expected_length = chrsize * (chrsize + 1) // 2

    A = np.zeros((cell_num, chrsize, chrsize), dtype=B.dtype)
    ptr = 0
    for k in range(chrsize):
        length_k = chrsize - k
        B_sub = B[:, ptr:ptr + length_k] 
        r = np.arange(length_k)
        c = r + k
        cell_idxs = np.arange(cell_num)[:, None]
        row_idxs = r[None, :]
        col_idxs = c[None, :]
        A[cell_idxs, row_idxs, col_idxs] = B_sub
        ptr += length_k
    return A


def coo_full(A_,rm_diag=True):
    if not isinstance(A_,np.ndarray):
        A_ = np.array([A_])

    _A = []
    for A in A_:
        if rm_diag:
            diag_full = np.zeros(A.shape[0], dtype=A.dtype)
            mask = A.row == A.col
            diag_full[A.row[mask]] = A.data[mask]
            D = sp.diags(diag_full, offsets=0, shape=A.shape, format='coo')
        else:
            D = 0

        result = A + A.T - D 
        _A.append(result.tocoo())
    
    return np.array(_A)


def dense_full(A_,rm_diag=True):
    n, m, m_check = A_.shape
    symmetric_matrices = A_ + A_.transpose((0, 2, 1))
    if not rm_diag:
        return symmetric_matrices
    original_diagonals = np.diagonal(A_, axis1=1, axis2=2)
    symmetric_matrices[:, np.arange(m), np.arange(m)] = original_diagonals
    
    return symmetric_matrices


def neighbor_ave_gpu(A, pad, return_Tensor=True, device='cpu', row_stride=1, col_stride=1):
    device = torch.device(device)

    if A.ndim == 2:
        A = A[None,:]

    if pad == 0:
        if return_Tensor:
            return torch.from_numpy(A).float().to(device)
        else:
            return A

    ll = pad * 2 + 1
    conv_filter = torch.ones(1, 1, ll, ll, dtype=torch.float32).to(device)
    B = F.conv2d(torch.from_numpy(A[:, None, :]).float().to(device), conv_filter, stride=(row_stride, col_stride), padding=pad * 2)
    B = B.squeeze(1)
    B = B[:, pad:-pad, pad:-pad] / float(ll * ll)

    if return_Tensor:
        return B
    else:
        return B.cpu().numpy()


def random_walk_gpu(A_batch, rp, return_Tensor=True, device='cpu'):
    device = torch.device(device)
    
    if not isinstance(A_batch, torch.Tensor):
        A_batch = torch.tensor(A_batch)
    
    A_batch = A_batch.float().to(device)
    
    if A_batch.dim() == 2:
        A_batch = A_batch.unsqueeze(0)
    batch_size, ngene, _ = A_batch.shape
    
    diag_mask = torch.ones((ngene, ngene), device=device)
    diag_mask = diag_mask - torch.diag(torch.diag(diag_mask))
    A_batch = A_batch * diag_mask
    
    col_sum = torch.sum(A_batch, dim=1)
    diag_mask = (col_sum == 0).float()
    diag_matrix = torch.diag_embed(diag_mask)
    A_batch = A_batch + diag_matrix
    
    Q_list = []

    stop_n = []
    
    for A in A_batch:
        P = torch.div(A, torch.sum(A, 0))
        Q = torch.eye(ngene, device=device)
        I = torch.eye(ngene, device=device)
        
        for _ in range(100):
            Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
            delta = torch.norm(Q - Q_new, 2)
            Q = Q_new
            if delta < 1e-6:
                stop_n.append(_)
                break
        
        Q_list.append(Q)

    # print(f'stop_n: len{len(stop_n)}, mean:{np.mean(stop_n):.1f}, max:{np.max(stop_n)}, min:{np.min(stop_n)}')
    
    Q_batch = torch.stack(Q_list, dim=0)
    

    if return_Tensor:
        return Q_batch
    else:
        return Q_batch.cpu().numpy()


def schicluster_gpu(A_batch, pad=1, rp=0.5, return_Tensor=False, device='cpu', with_log=False, batch_size=2000):
    if with_log:
        A_batch = np.log1p(A_batch)
        print('schicluster with log1p')
    
    if A_batch.ndim == 2:
        A_batch = A_batch[None, :]
        reshape = True
    else:
        reshape = False

    def process_core(sub_batch):
        A = neighbor_ave_gpu(sub_batch, pad, device=device)
        if rp == -1:
            Q_sub = A[:]
        else:
            Q_sub = random_walk_gpu(A, rp, device=device)
        return Q_sub

    N = A_batch.shape[0]

    if N <= batch_size:
        Q = process_core(A_batch)
    else:
        print(f'Input size {N} > {batch_size}, imputing in batches...')
        output_list = []
        for i in range(0, N, batch_size):
            sub_batch = A_batch[i : i + batch_size]
            sub_Q = process_core(sub_batch)
            if not return_Tensor:
                sub_Q = sub_Q.cpu()
            
            output_list.append(sub_Q)
        
        Q = torch.cat(output_list, dim=0)

    if reshape:
        Q = Q[0]

    if return_Tensor:
        return Q 
    else:
        if isinstance(Q, torch.Tensor):
            return Q.cpu().numpy()
        return Q
    

def coo_dense_exchange(A, to_which=None):
    if to_which is not None:
        assert to_which in ['coo', 'dense'], "to_which must be 'coo' or 'dense'"
    if A.ndim == 1 and isinstance(A[0], scipy.sparse._coo.coo_matrix):
        if to_which == 'coo':
            return A
        return np.array([matrix.toarray().astype('float32') for matrix in A])
    elif A.ndim == 3:
        if to_which == 'dense':
            return A
        return np.array([coo_matrix(matrix) for matrix in A])
    else:
        assert 'input A has wrong shape or type!'

def get_head_len(size, n):
    if n==-1 or n > size:
        n = size
    head_len = int(((size*2-n+1)*n)/2)
    tail_len = int((size+1)*size//2) - head_len
    return head_len, tail_len


def classify(labels,
             batch_size=32,
             cell_init=None,
             cell_encoder=None,
             seed=None,
             no_tqdm=True):
    
    from Modules import FeatureEncoder, MultiClassFocalLossWithAlpha, MultiClassifier

    available_devices = get_available_gpus()
    if not available_devices:
        print("Warning: No GPU found. Running on CPU.")
        device = torch.device('cpu')
    else:
        gpu_id = available_devices[0]
        device = torch.device(f'cuda:{gpu_id}')

    cell_encoder.float().to(device)
    X = torch.from_numpy(cell_init).float().to(device)
    raw_labels = labels

    n_samples = cell_init.shape[0]
    cell_encoder.eval()
    with torch.no_grad():
        ttt = cell_encoder(X[:2,:])
        input_dim = ttt.shape[-1]

    num_classes = len(np.unique(raw_labels))

    le = LabelEncoder()
    integer_labels = le.fit_transform(raw_labels) 
    y = torch.tensor(integer_labels, dtype=torch.long, device=device)

    counts_tensor = torch.bincount(y)
    counts_list = counts_tensor.cpu().tolist()

    indices = np.arange(n_samples)
    
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=y.cpu().numpy(), random_state=seed
    )
    X_train, y_train = X[train_indices], y[train_indices]
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_test, y_test = X[test_indices], y[test_indices]
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    cell_encoder1 = copy.deepcopy(cell_encoder)

    mlp = FeatureEncoder(input_seq_len=input_dim, 
                            conv_channels=[32,32], 
                            linear_features=[num_classes],
                            kernel_size=7,
                            activation=nn.ReLU,
                            last=True,
                            pool=False,
                            dp=None)
    
    model = MultiClassifier(cell_encoder1, mlp).to(device, non_blocking=True)
    params_to_optimize = model.set_lr([1e-3, 1e-3])
    optimizer = optim.Adam(params_to_optimize)
    
    
    def get_adaptive_gamma(labels):
        counts = np.bincount(labels)
        counts = counts[counts > 0]
        if len(counts) == 0:
            return 2.0
        n_max = np.max(counts)
        n_min = np.min(counts)
        imbalance_ratio = n_max / max(1, n_min)
        raw_gamma = np.log(imbalance_ratio)
        gamma = max(1.0, raw_gamma)
        gamma = min(gamma, 5.0)
        return gamma

    gamma = get_adaptive_gamma(y_train.detach().cpu().numpy())
    criterion = MultiClassFocalLossWithAlpha(alpha=counts_list, gamma=gamma, device=device)

    best_train_loss = float('inf') 
    epochs_no_improve = 0 
    best_model_weights = copy.deepcopy(model.state_dict()) 

    pbar = trange(1, 500 + 1, disable=no_tqdm, desc="Training Classifier")
    for epoch in pbar:
        model.train()

        epoch_losses, epoch_train_preds, epoch_train_labels = [], [], []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            epoch_train_preds.append(torch.argmax(logits.detach(), dim=1).cpu())
            epoch_train_labels.append(batch_y.cpu())
        
        avg_loss = np.mean(epoch_losses)
        all_train_labels = torch.cat(epoch_train_labels).numpy()
        all_train_preds = torch.cat(epoch_train_preds).numpy()

        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_bacc = balanced_accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        
        desc_str = f"LOSS:{avg_loss*1e4:.2f} | ACC:{train_acc:.3f} | BACC: {train_bacc:.3f} | F1: {train_f1:.3f}"

        pbar.set_description(desc_str)

        # # early stop
        if avg_loss < best_train_loss - (1e-6):
            best_train_loss = avg_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 30:
            model.load_state_dict(best_model_weights)
            break

    model.eval()
        
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(batch_y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()


    acc = accuracy_score(all_labels, all_preds)
    real_label = all_labels
    test_label = all_preds
    micro_F1 = f1_score(real_label, test_label, average='micro')
    macro_F1 = f1_score(real_label, test_label, average='macro')
    micro_Precision = precision_score(real_label, test_label,
                                        average='micro')
    macro_Precision = precision_score(real_label, test_label,
                                        average='macro')
    micro_Recall = recall_score(real_label, test_label,
                                average='micro')
    macro_Recall = recall_score(real_label, test_label,
                                average='macro')
    bacc = balanced_accuracy_score(real_label, test_label)
    mcc = matthews_corrcoef(real_label, test_label)

    metrics = {
    'acc': acc,
    'micro_F1': micro_F1,
    'macro_F1': macro_F1,
    'micro_Precision': micro_Precision,
    'macro_Precision': macro_Precision,
    'bacc': bacc,
    'mcc': mcc,
    'micro_Recall': micro_Recall,
    'macro_Recall': macro_Recall
    }

    return metrics


def std_print(keys, datas, style=None):
    processed_datas = list(datas)
    if style and style.lower() == 'raw':
        processed_datas = [d * 100 for d in datas]

    formatted_datas = [f"{d:.2f}" for d in processed_datas]  
    col_widths = [max(len(k), len(d)) for k, d in zip(keys, formatted_datas)]
    separator_line = "+" + "+".join(['-' * (w + 2) for w in col_widths]) + "+"
    key_line = "|"
    for i, key in enumerate(keys):
        key_line += f" {key:^{col_widths[i]}} |"
    data_line = "|"
    for i, data_str in enumerate(formatted_datas):
        data_line += f" {data_str:^{col_widths[i]}} |"
    print(separator_line)
    print(key_line)
    print(separator_line)
    print(data_line)
    print(separator_line)


def loss_func(pred, target, temperature=1.0, target_power=1.0):
    log_probs = F.log_softmax(pred / temperature, dim=1)
    weighted_target = torch.sign(target) * torch.abs(target).pow(target_power)
    per_sample_loss = -torch.sum(weighted_target * log_probs, dim=1)
    loss = torch.mean(per_sample_loss)
    
    return loss / target.shape[-1]


def normalize_(A, config=None):
    if config and 'tsnm' in config and config['tsnm']:
        A = [torch.from_numpy(x).float() for x in A]
        A_norm = [F.softmax(x, dim=-1) for x in A]
    else:
        A_norm = [np.log2(x + 1.0) for x in A]
        A_norm = [torch.from_numpy(x).float() for x in A_norm]
        
    return A_norm

def print_scores(labels,cell_feats,to_dim=None, return_pred_label=False):
    if isinstance(cell_feats, np.ndarray):
        combined_feat = cell_feats
        if to_dim is not None:
            if to_dim == combined_feat.shape[-1]:
                return 0,0,0
            pca = PCA(n_components=to_dim)
            combined_feat = pca.fit_transform(combined_feat)
    else:
        reduce_feats = []
        for feats in cell_feats:
            if to_dim is not None:
                scaler1 = StandardScaler(with_mean=True, with_std=True)
                A_std = scaler1.fit_transform(feats)
                pca = PCA(n_components=to_dim)
                A_reduced = pca.fit_transform(A_std)
                reduce_feats.append(A_reduced)
            else:
                reduce_feats.append(feats)
        combined_feat = np.concatenate(reduce_feats,axis=1)

    print(f'feat_shape:{combined_feat.shape} label_shape:{labels.shape}')

    spectral_clustering = SpectralClustering(n_clusters=len(np.unique(labels)), affinity='nearest_neighbors', n_init=200)
    pred_label = spectral_clustering.fit_predict(combined_feat)
    ari = adjusted_rand_score(labels, pred_label)
    nmi = normalized_mutual_info_score(labels, pred_label)
    hm = homogeneity_score(labels, pred_label)
    fm = fowlkes_mallows_score(labels, pred_label)
    ami = adjusted_mutual_info_score(labels, pred_label)
    asw = silhouette_score(combined_feat, pred_label, metric='cosine')
    std_print(['ARI','NMI','HM','FM', 'AMI', 'ASW'],[ari,nmi,hm,fm,ami,asw],style='raw')
    
    if return_pred_label:
        return combined_feat, ari, acroc, pred_label
    else:
        return combined_feat, ari, acroc

def save_vis(cell_labels, cell_embeddings, config=None, batchinfo=None):
    _, ari, acroc = print_scores(cell_labels, cell_embeddings)
    
def run_train(config, train=True, no_tqdm=False, return_batchinfo=False):
    from train import Train
    trainer = Train(config, no_tqdm=no_tqdm)

    trainer.prep_data()
    trainer.prep_model(train_all=False)

    if train:
        trainer.train()
        trainer.joint_train()

        trainer.save_model()

    else:
        trainer.load_model()

    features = trainer.fit()
    labels = trainer.labels
    batchinfo = trainer.batchinfo
 
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    if return_batchinfo:
        return features, labels, batchinfo
    else:
        return features, labels

def set_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def run_impute(config, train=True, pad=0, row_stride=1, col_stride=1):
    from train import Train
    trainer = Train(config, mode='impute', no_tqdm=False)

    trainer.prep_data_impute()
    trainer.prep_model_impute()

    if train:
        trainer.train_impute()
        trainer.save_model_impute()

    else:
        trainer.load_model_impute()

    cell_feats = trainer.fit_impute_cell_feat()
    raw_matrices = trainer.raw_matrices_numpy
    imputed_matrices = trainer.fit_impute(pad=pad, row_stride=row_stride, col_stride=col_stride)
    labels = trainer.labels

    return cell_feats, raw_matrices, imputed_matrices, labels

def analyze_cluster_composition(true_labels, cluster_labels):
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    composition_details = defaultdict(lambda: defaultdict(dict))
    
    unique_clusters = np.unique(cluster_labels)

    for cluster_id_np in unique_clusters:
        cluster_id = int(cluster_id_np)

        indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
        original_labels_in_cluster = true_labels[indices_in_cluster]
        
        unique_true_labels_here = np.unique(original_labels_in_cluster)
        
        for true_label in unique_true_labels_here:
            mask = (original_labels_in_cluster == true_label)
            specific_indices = indices_in_cluster[mask]
            count = len(specific_indices)
            
            composition_details[cluster_id][true_label] = {
                'count': count,
                'indices': specific_indices.tolist()
            }
    
    ana_dict = dict(composition_details)
    print("-"*30)
    for cluster_id, composition in sorted(ana_dict.items()):
        print(f"\n[ Cluster {cluster_id} ]:")
        print("-" * 30)
        for true_label, details in sorted(composition.items()):
            count = details['count']
            if count < 10:
                continue
            indices = details['indices']
            print(f"  Real cluster: '{true_label}'", end='\t')
            print(f"  Counts: {count}")
            
    return ana_dict

def SICP_BCP(cm: sp.coo_matrix, chr_len, mode:str, batch_size=1000, device='cpu'):
    my_device = torch.device(device) 
    features = []
    num_cells = len(cm) 
    for start in range(0, num_cells, batch_size):
        end = min(start + batch_size, num_cells)
        batch_adj_list = cm[start:end]  
        
        batch_dense = np.array([
            np.resize(my_cell_adj.toarray(), (chr_len, chr_len)) 
            if my_cell_adj.shape != (chr_len, chr_len) else my_cell_adj.toarray()
            for my_cell_adj in batch_adj_list
        ])

        batch_dense = torch.tensor(batch_dense, dtype=torch.float32, device=my_device)  # [batch_size, chr_len, chr_len]

        if mode.upper() == 'SICP':
            batch_feat = Small_Domain_Struct_Contact_pro_batch(
                contact_matrix=batch_dense, index=chr_len, scale=1
            ).cpu().numpy() 

        elif mode.upper() == 'BCP':
            batch_feat = Bin_contact_pro_batch(
                contact_matrix=batch_dense, index=chr_len
            ).cpu().numpy()
        
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'SICP' and 'BCP'.")

        features.append(batch_feat)

        del batch_dense
        del batch_feat
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    features = np.concatenate(features, axis=0)

    return features

def calculate_chr_bins(genome_reference_path, resolution, chr_order):
    chrom_sizes = {}
    with open(genome_reference_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                chrom, length = parts
                # chrom_sizes[chrom] = int(length)
                chrom_sizes[chrom] = math.ceil(int(length) / resolution)

    bin_counts = []
    for chrom in chr_order:
        if chrom not in chrom_sizes:
            raise ValueError(f"Chromosome '{chrom}' not found in sizes file: {genome_reference_path}")
        
        length = chrom_sizes[chrom]
        # bins = math.ceil(length / resolution)
        bins = length
        bin_counts.append(int(bins))

    return bin_counts, chrom_sizes  

def get_chrsizes(genome_reference_path, resolution, chrom_list):
    
    if not os.path.exists(genome_reference_path):
        raise FileNotFoundError(f"Genome sizes file not found at: {genome_reference_path}\nPlease update the path in GENOME_SIZES_PATHS.")

    chr_lens, chrom_sizes = calculate_chr_bins(genome_reference_path, resolution, chrom_list)

    return chr_lens, chrom_sizes

def slice_coo(coo, a, b):
    csr = coo.tocsr()
    sub_matrix_csr = csr[a:b, a:b]

    return sub_matrix_csr.tocoo().toarray()

def slice_coo_batch(A, a, b):
    dense_slices = [slice_coo(coo,a,b) for coo in A]
    return np.array(dense_slices)

def format_number(num, precision=1):
    suffixes = ['', 'k', 'M']

    import math
    magnitude_index = int(math.floor(math.log(num, 1000)))
    
    magnitude_index = min(magnitude_index, len(suffixes) - 1)

    scaled_num = num / (1000 ** magnitude_index)

    formatter = f"{{:.{precision}f}}"
    formatted_num_str = formatter.format(scaled_num)

    if '.' in formatted_num_str:
        formatted_num_str = formatted_num_str.rstrip('0').rstrip('.')
        
    return f"{formatted_num_str}{suffixes[magnitude_index]}"

def load_bulk(cell_type, depth, resolution, chr_id=1, bound=None, dir_path='/liudongliang/bulk_hic'):
    import cooler

    chr_id = chr_id+1
    if chr_id == 23:
        chr_id = 'X'

    file_name = {
                'K562':
                {
                    'low': 'K562_low_4DNFIYIJQEYA',
                    'deep': 'K562_deep_4DNFI18UHVRO'
                },
                'GM12878':
                {
                    'low': 'GM12878_low_4DNFIULI6AAA',
                    'deep': 'GM12878_deep_4DNFIXP4QG5B'
                },
                'HAP1':
                {
                    'low': 'HAP1_low_4DNFI6NR6Y35',
                    'deep': 'HAP1_deep_4DNFINSKEZND'
                },
                'HeLa':
                {
                    'low': 'HeLa_4DNFIBM9QCFG',
                    'deep': 'HeLa_4DNFIBM9QCFG'
                }
                }
    matrix_path = f"{dir_path}/{file_name[cell_type][depth]}::/resolutions/{int(resolution)}"
    c = cooler.Cooler(matrix_path)
    mat_sparse = c.matrix(balance=False, sparse=True).fetch(f'chr{chr_id}')
    mat_dense = mat_sparse.toarray().astype(np.float32)
    if bound is None:
        return mat_dense
    else:
        start = int(bound[0] // resolution)
        end = int(bound[1] // resolution)
        return mat_dense[start:end, start:end]


def fit_size(A_list:list):
    min_size = 1e6
    B_list = []
    for A in A_list:
        if A.shape[-1] < min_size:
            min_size = A.shape[-1]
    for A in A_list:
        B = A[:,:min_size,:min_size]
        B_list.append(B)
    
    return B_list


def cal_cor(scmat, bulkmat, flatten_mode='norm', mode='spearman'):
    from scipy.stats import pearsonr, spearmanr
    if mode == 'spearman':
        cor_func = spearmanr
    elif mode == 'pearson':
        cor_func = pearsonr
    score_list = []
    if flatten_mode == 'norm':
        bulkm = bulkmat.reshape((-1))
    else:
        bulkm = flatten_ndiags(bulkmat, -1)
    for mat in scmat:
        if np.all(mat == 0):
            print('Warning: find an all zore imputed matrix!')
            score_list.append(0)
            continue
        if flatten_mode == 'norm':
            scm = mat.reshape((-1))
        else:
            scm = flatten_ndiags(mat, -1)
        score_list.append(cor_func(scm, bulkm)[0])
    return np.array(score_list)


def cal_cor_vectorized(scmat, bulkmat, flatten_mode='norm', mode='spearman'):
    n_cells = scmat.shape[0]

    if flatten_mode == 'norm':
        sc_vectors = scmat.reshape(n_cells, -1)
        bulk_vector = bulkmat.ravel()
    else:
        try:
            sc_vectors = flatten_ndiags(scmat, -1)
            bulk_vector = flatten_ndiags(bulkmat, -1)
        except TypeError:
            print("Warning: flatten_ndiags not vectorized. Falling back to slower loop or consider parallelization.")
            return cal_cor(scmat, bulkmat, flatten_mode, mode)

    if mode == 'spearman':
        sc_vectors = rankdata(sc_vectors, axis=1)
        bulk_vector = rankdata(bulk_vector)

    sc_mean = sc_vectors.mean(axis=1, keepdims=True)
    bulk_mean = bulk_vector.mean()
    
    sc_centered = sc_vectors - sc_mean
    bulk_centered = bulk_vector - bulk_mean

    numerator = np.dot(sc_centered, bulk_centered)

    sc_std_dev = np.sqrt(np.sum(sc_centered**2, axis=1))
    bulk_std_dev = np.sqrt(np.sum(bulk_centered**2))
    
    denominator = sc_std_dev * bulk_std_dev

    scores = np.zeros_like(denominator)
    valid_mask = denominator != 0
    scores[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    is_zero_mat = np.all(scmat == 0, axis=(1, 2))
    scores[is_zero_mat] = 0.0
    
    if np.any(is_zero_mat):
        print(f'Warning: Found {np.sum(is_zero_mat)} all-zero imputed matrices. Their scores are set to 0.')

    return scores


def parse_chrom(s):
    val = s.replace('chr', '')
    mapping = {'X': 23, 'Y': 24}

    return mapping.get(val) or int(val)


def std_print_cmp(scores, chrom_list, pos_dict, cell_type=None):
    if cell_type is None:
        title = "Mean Correlation Scores (All Cells)"
        pos = slice(None)
    else:
        if cell_type not in pos_dict:
            print(f"Error: Cell type '{cell_type}' not found in pos_dict.")
            return
        title = f"Mean Correlation Scores (Filtered by Cell Type: {cell_type})"
        pos = pos_dict[cell_type]

    methods = list(scores.keys())
    header = [cell_type if cell_type is not None else ''] + methods + ['Best (vs 2nd)']
    
    col_widths = [len(h) for h in header]
    col_widths[0] = max(col_widths[0], 12)
    for i in range(1, len(methods) + 1):
        col_widths[i] = max(col_widths[i], 8)
    col_widths[-1] = max(col_widths[-1], 20)

    print(f"\n--- {title} ---")
    
    header_line = "".join([f"{h:<{col_widths[i]}} | " for i, h in enumerate(header)])
    print(header_line)
    
    separator_line = "".join(['-' * w + ' | ' for w in col_widths])
    print(separator_line)

    for chrom in chrom_list:
        row_scores = {}
        for method in methods:
            chrom_scores = scores[method].get(chrom, np.array([]))[pos]
            if chrom_scores.size > 0:
                row_scores[method] = np.mean(chrom_scores)
            else:
                row_scores[method] = np.nan 

        valid_scores = sorted(
            [(score, method) for method, score in row_scores.items() if not np.isnan(score)],
            key=lambda item: item[0],
            reverse=True
        )

        best_col_str = ""
        if len(valid_scores) >= 2:
            best_score, best_method = valid_scores[0]
            second_best_score, _ = valid_scores[1]
            diff = best_score - second_best_score
            best_col_str = f"{best_method} (+{diff:.3f})"
        elif len(valid_scores) == 1:
            _, best_method = valid_scores[0]
            best_col_str = f"{best_method} (only one)"
        else:
            best_col_str = "N/A"

        row_line = f"{chrom:<{col_widths[0]}} | "
        for i, method in enumerate(methods):
            score = row_scores[method]
            score_str = f"{score:.3f}" if not np.isnan(score) else "N/A"
            row_line += f"{score_str:<{col_widths[i+1]}} | "

        row_line += f"{best_col_str:<{col_widths[-1]}} | "
        print(row_line)
    
    print("-" * len(separator_line))
    print()




if __name__ == '__main__':
    pass