from utils import *
from Process import *
from Modules import *
from train import *
from visualization import *
import cooler

class MultiImputor():
    def __init__(self, config):
        
        self.config = config
        self.bound = config['bound']
        self.resolution = config['impute_res']

        if self.bound is not None:
            self.start = int(self.bound[0] // self.resolution)
            self.end = int(self.bound[1] // self.resolution)
        
        self.chrom_list = config['chrom_list']
        self.impute_chrom_list = config['impute_chrom_list']

        self.imputed_dict = {}
        self.bulk = {}
        self.cor_scores = {}
    

    def train_impute(self):
        run_impute(self.config,train=True)
        
    
    def load_from_model(self, pad=1, row_stride=1, col_stride=1):
        _, raw_matrices, imputed_matrices, self.labels = run_impute(self.config,train=False, pad=pad, row_stride=row_stride, col_stride=col_stride)

        raw_dict = {}
        my_impute = {}
        for chr_id, raw, imputed in zip(self.impute_chrom_list, raw_matrices, imputed_matrices):
            raw_dict[chr_id] = raw
            imputed = (neighbor_ave_gpu(raw + 0.5, return_Tensor=False, device='cuda:0', pad=1)) * imputed
            my_impute[chr_id] = imputed
            
        self.imputed_dict['raw'] = raw_dict
        self.imputed_dict['mine'] = my_impute
            
            
    def load_from_file(self, file_paths:list, names:list):
        for name, file_path in zip(names, file_paths):
            with open(file_path,'rb') as f:
                imputed_ = pickle.load(f)
            self.imputed_dict[name] = imputed_
            print(f'load {name} imputed from {file_path}')
            

    def load_bulk(self, bulk_dir, cell_types:list=None, depth='low'):
        if cell_types is None:
            cell_types = np.unique(self.labels)
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
        
        for cell_type in cell_types:
            self.bulk[cell_type] = {}
            matrix_path = f"{bulk_dir}/{file_name[cell_type][depth]}.mcool::/resolutions/{int(self.resolution)}"
            chrs_one_type = cooler.Cooler(matrix_path)
            for chrom in self.impute_chrom_list:
                mat_sparse = chrs_one_type.matrix(balance=False, sparse=True).fetch(chrom)
                if self.bound is None:
                    mat_dense = mat_sparse.toarray().astype(np.float32)
                    mat_dense = mat_dense[None, :]
                else:
                    print(f'loading {cell_type} bulk part: [{self.start}:{self.end}]')
                    mat_dense = slice_coo(mat_sparse, self.start, self.end)
                self.bulk[cell_type][chrom] = mat_dense

            print(f'load {cell_type} bulk data from {matrix_path}')


    def fetch_type_pos(self, labels=None):
        if labels is None:
            labels = self.labels
        self.pos_dict = {}
        for cell_type in np.unique(labels):
            pos = np.where(labels == cell_type)[0]
            self.pos_dict[cell_type] = pos


    def cal_cor(self, labels=None, mode='spearman', rewrite=False, add_pseudo_bulk=False):
        if labels is None:
            labels = self.labels
        
        self.cor_scores[mode] = {}
        for method in self.imputed_dict.keys():
            try:
                if method in self.cor_scores[mode].keys() and not rewrite:
                    print(f'skip calculating {method} {mode} score')
                    continue
            except:
                pass
            print(f'{method} method calculating cor score')
            self.cor_scores[mode][method] = {}
            for chrom in self.impute_chrom_list:
                chr_score = np.zeros_like(labels, dtype=float)
                imputed = self.imputed_dict[method][chrom]
                for cell_type in self.pos_dict.keys():
                    type_pos = self.pos_dict[cell_type]
                    bulk = self.bulk[cell_type][chrom]
                    imputed_one_type = imputed[type_pos]
                    imputed_one_type, bulk = fit_size([imputed_one_type, bulk])
                    if 'higashi' in method.lower():
                        bulk_mat = bulk[0]
                    elif method.lower() == 'schicluster':
                        bulk_mat = np.log2(bulk[0] + 1)
                    else:
                        bulk_mat = np.log1p(bulk[0])
                    chr_score[type_pos] = cal_cor_vectorized(imputed_one_type, bulk_mat, 'norm', mode)
                self.cor_scores[mode][method][chrom] = chr_score

        if add_pseudo_bulk:
            print('calculating pseudo bulk cor scores')
            self.cor_scores[mode]['pseudo_bulk'] = {}
            for chrom in self.impute_chrom_list:
                chr_score = np.zeros_like(labels, dtype=float)
                for cell_type in self.pos_dict.keys():
                    type_pos = self.pos_dict[cell_type]
                    bulk = self.bulk[cell_type][chrom]
                    imputed_one_type = self.imputed_dict['raw'][chrom][type_pos]
                    pseudo_bulk = np.mean(imputed_one_type, axis=0, keepdims=True)
                    pseudo_bulk, bulk = fit_size([pseudo_bulk, bulk])
                    chr_score[type_pos] = cal_cor_vectorized(pseudo_bulk, np.log1p(bulk[0]), 'norm', mode)
                self.cor_scores[mode]['pseudo_bulk'][chrom] = chr_score


    def avg_conv(self, methods:list, pad=1):
        for method in methods:
            new_method = f'{method}_pad{pad}'
            self.imputed_dict[new_method] = {}
            for chrom in self.imputed_dict[method].keys():
                matrices = self.imputed_dict[method][chrom]
                smatrices = schicluster_gpu(matrices, pad=pad, rp=-1, return_Tensor=False, device='cuda:0')
                self.imputed_dict[new_method][chrom] = smatrices

            print(f'smooth {method} to a new {new_method}')


    def draw_bin(self):
        pass

    def compare_whole(self, chr_id=0, methods='all', cell_type=None, show_num=5, cids=None, cor='spearman'):
        if isinstance(chr_id, int):
            chrom = self.impute_chrom_list[chr_id]
        elif isinstance(chr_id, str):
            chrom = chr_id
        else:
            raise TypeError("chr_id must be an integer index or a string name.")
        
        if chrom not in self.impute_chrom_list:
            print(f"{chrom} is not in chrom list:{self.impute_chrom_list}")
            return 
        
        print(f"Displaying chromosome: {chrom}")
        if cids is None:
            if cell_type is None:
                cids = np.random.randint(0, len(self.labels), show_num)
            else:
                pos = self.pos_dict[cell_type]
                idx = np.random.randint(0, len(pos), show_num)
                cids = pos[idx]
        else:
            show_num = len(cids)
        print(cids)
        if methods == 'all':
            methods = [k for k in self.imputed_dict.keys()] + ['bulk']
        if chrom == 'chr1':
            rm_gap_start = 120
            rm_gap_end = 150
        elif chrom == 'chr3':
            rm_gap_start = 88
            rm_gap_end = 95
        data_list = []
        scores_list = []
        q_list = []
        for method in methods:
            if method != 'bulk':
                p2show = self.imputed_dict[method][chrom][cids]
                p2show = remove_gap(p2show, start=rm_gap_start, end=rm_gap_end)
                # if method == 'mine':
                #     raw = data_list[0]
                #     p2show = (neighbor_ave_gpu(raw + 0.5, return_Tensor=False, device='cuda:0', pad=1)) * p2show
                data_list.append(p2show)
                scores_list.append(self.cor_scores[cor][method][chrom][cids])
                q_list.append(0.6 if method == 'raw' else 0.95)

            else:
                t_data = []
                for i in cids:
                    p2show = self.bulk[self.labels[i]][chrom][0]
                    p2show = remove_gap(p2show, start=rm_gap_start, end=rm_gap_end)
                    t_data.append(p2show)
                data_list.append(np.stack(t_data, axis=0))
                scores_list.append(np.full(show_num, 1.0))
                q_list.append(0.95)

        show_comparison(data_list, methods, scores_list, q_list, self.labels[cids], dpi=600, save_path=f'../imputation/{chrom}_comparison.png')


    def compare_part(self, chr_id=0, cell_types:list=None, cell_num=10, norm_func=lambda x: x):
        if isinstance(chr_id, int):
            chrom = self.impute_chrom_list[chr_id]
        elif isinstance(chr_id, str):
            chrom = chr_id
        else:
            raise TypeError("chr_id must be an integer index or a string name.")
        
        if chrom not in self.impute_chrom_list:
            print(f"{chrom} is not in chrom list:{self.impute_chrom_list}")
            return 
        
        print(f"Displaying chromosome: {chrom}")
        all_cells = [self.pos_dict[ct] for ct in cell_types]
        for i, c in enumerate(all_cells):
            if len(c) < cell_num:
                print(f'Warning: {cell_types[i]} only has only {len(c)} cells less than expected {cell_num}')
        idces = [np.random.randint(0, len(c), cell_num) for c in all_cells]
        selected_cells = [x[i] for x,i in zip(all_cells, idces)]

        raws = [self.imputed_dict['raw'][chrom][i] for i in selected_cells]
        imputes = [self.imputed_dict['mine'][chrom][i] for i in selected_cells]
        raw_sum = np.array([np.mean(norm_func(x), axis=0) for x in raws])
        imputed_sum = np.array([np.mean(norm_func(x), axis=0) for x in imputes])
        bulk = np.array([norm_func(self.bulk[ct][chrom]) for ct in cell_types])

        methods = ['raw', 'mine', 'bulk']
        data_list = [raw_sum, imputed_sum, bulk]
        scores_list = None
        q_list = [0.6,0.95,0.95]

        return data_list, methods, scores_list, q_list, cell_types

    def sub4compare(self):
        pass

    def normalization(self):
        pass
  
  
