import sys
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from utils import *
from plugins import *
from Modules import *


class Processor:
    def __init__(self,config=None,no_tqdm=False):
        self.config = config
        self.no_tqdm = no_tqdm
        self.dataset = config['dataset']
        self.origin_dir = config['data_dir']
        self.resolution = config['resolution']
        self.chrom_list = config['chrom_list']
        res_str = format_number(self.resolution)
        
        self.memory_limit = True if self.resolution < 5e5 else False
        
        try:
            self.labels = np.load(os.path.join(self.origin_dir,self.dataset+res_str,'label_info.npy'))
        except:
            self.labels = np.load(os.path.join(self.origin_dir,self.dataset,'label_info.npy'))
        
        try:
            try:
                self.batchinfo = np.load(os.path.join(self.origin_dir,self.dataset+res_str,'batch_info.npy'))
            except:
                self.batchinfo = np.load(os.path.join(self.origin_dir,self.dataset,'batch_info.npy'))
        except:
            self.batchinfo = None
            
        try:
            self.data_path = os.path.join(self.origin_dir,self.dataset+res_str,'data.npy')
            self.weight_path = os.path.join(self.origin_dir,self.dataset+res_str,'weight.npy')
            self.data = np.load(self.data_path)
            self.weight = np.load(self.weight_path)
            print(f'processor load data from:{os.path.join(self.origin_dir,self.dataset+res_str)}')
            self.dataset_dir = os.path.join(self.origin_dir,self.dataset+res_str)
            
        except:
            self.data_path = os.path.join(self.origin_dir,self.dataset,'data.npy')
            self.weight_path = os.path.join(self.origin_dir,self.dataset,'weight.npy')
            self.data = np.load(self.data_path)
            self.weight = np.load(self.weight_path)
            print(f'expect resolution {res_str}, but not found, processor try default path:{os.path.join(self.origin_dir,self.dataset)}')
            self.dataset_dir = os.path.join(self.origin_dir,self.dataset)
            
        self.cell_num = np.max(self.data[:, 0]) + 1
        self.chr_num = len(self.chrom_list)

        self.chrsizes, self.chrom_sizes = get_chrsizes(config['genome_reference_path'], self.resolution, self.chrom_list)
        # ===================for unexcepted condition=========================
        lens_file_path = os.path.join(self.dataset_dir, 'chr_lens.npy')
        if os.path.exists(lens_file_path):
            print(f"chr_lens file exist: {lens_file_path}")
            loaded_lens = np.load(lens_file_path).astype(int).tolist()
            
            if loaded_lens != self.chrsizes:
                print("\n  (Dimension Mismatch Detected)!")
                print(f"   - (Calculated from Ref): {self.chrsizes}")
                print(f"   - (Loaded from scool):   {loaded_lens}")
                print(f"   -> using loaded chr_lens...")

                self.chrsizes = loaded_lens

                if len(loaded_lens) != len(self.chrom_list):
                     raise ValueError(f"Error: chr num in chr_lens.npy ({len(loaded_lens)}) doesn't match chrom_list ({len(self.chrom_list)}) ")
                
                for idx, chrom in enumerate(self.chrom_list):
                    self.chrom_sizes[chrom] = loaded_lens[idx]
                
                print("chr_lens is overwritten")
            else:
                print("Ref is same with Given File")
        else:
            print(f"can't find {lens_file_path}, using Ref to calculate")
        # ===============================================================

        self.start_pos = [np.sum(self.chrsizes[:i]) for i in range(len(self.chrsizes))]

        """
        data: a 4-col numpy array  [cell_id  chr_id  bin1  bin2] 
        weight: corresponding contacts
        """
        self.statistic_dict = None
        self.contact_matrices_coo = {}
        self.imputed_cm_coo = {}
        self.chr_feats_dict = {} # {chr_id: {feat_mode: features}}
        self.available_devices = get_available_gpus()
        self.default_gpu_id = self.available_devices[0]
        self.default_device = torch.device(f'cuda:{self.default_gpu_id}')

        self.get_cm4chr_all()


    def calculate_hic_stats(self) -> pd.DataFrame:
        hic_data_list = [coo_dense_exchange(x, 'dense') for x in self.contact_matrices_coo]
        cell_num = self.cell_num

        total_intra_contacts = np.zeros(cell_num, dtype=np.float64)
        non_zero_counts = np.zeros(cell_num, dtype=np.int64)

        for i, chr_matrix in enumerate(hic_data_list):
            contacts_this_chr = np.sum(chr_matrix, axis=(1, 2))
            non_zeros_this_chr = np.count_nonzero(chr_matrix, axis=(1, 2))

            total_intra_contacts += contacts_this_chr
            non_zero_counts += non_zeros_this_chr

        # mydebug(total_intra_contacts, False)
        # mydebug(non_zero_counts)

        total_possible_bins = sum([size**2 for size in self.chrsizes])
        sparsity = (non_zero_counts / total_possible_bins)

        stats_df = pd.DataFrame({
            'total_intra_contacts': total_intra_contacts,
            'non_zero_counts': non_zero_counts,
            'sparsity': sparsity
        })
        
        stats_df.index.name = 'cell_id'

        return stats_df

    
    def normalize_contact_matrices(self):
        statistic_dict = self.calculate_hic_stats()
        contact_matrices = [coo_dense_exchange(x, 'dense') for x in self.contact_matrices_coo]
        total_intra_contacts = np.array(statistic_dict['total_intra_contacts'])
        delta = total_intra_contacts/np.mean(total_intra_contacts) + 1e-10
        normalized_contact_matrices = [x / delta[:, np.newaxis, np.newaxis] for x in contact_matrices]
        self.contact_matrices_coo = [coo_dense_exchange(x, 'coo') for x in normalized_contact_matrices]
        # mydebug(normalized_contact_matrices)


    def gen_extra_feats(self):
        """
        the feat should be (626, n)
        """
        hic_data_list = [coo_dense_exchange(x, 'dense') for x in self.contact_matrices_coo]
        cell_num = self.cell_num
        chr_num = self.chr_num

        extra_feat_list = []

        each_intra_contacts = np.zeros((chr_num,cell_num))
        non_zero_counts = np.zeros((chr_num,cell_num))

        for i, chr_matrix in enumerate(hic_data_list):
            contacts_this_chr = np.sum(chr_matrix, axis=(1, 2))
            non_zeros_this_chr = np.count_nonzero(chr_matrix, axis=(1, 2))

            each_intra_contacts[i] += contacts_this_chr
            non_zero_counts[i] += non_zeros_this_chr

        each_intra_contacts = each_intra_contacts.T
        non_zero_counts = non_zero_counts.T

        BCP_feats = []
        SICP_feats = []
        for c in tqdm(range(chr_num), desc='gen BCP and SICP feats', disable=self.no_tqdm):
            BCP_feats.append(SICP_BCP(self.contact_matrices_coo[c], 'cpu', self.chrsizes[c], 'BCP'))
            SICP_feats.append(SICP_BCP(self.contact_matrices_coo[c], 'cpu', self.chrsizes[c], 'BCP'))
        
        BCP_feats = np.concatenate(BCP_feats, axis=1)
        SICP_feats = np.concatenate(SICP_feats, axis=1)

        extra_feat_list.append(BCP_feats)

        extra_feats = np.concatenate(extra_feat_list, axis=1)
        
        return extra_feats


    def statistic(self):
        """
        func:
            get infos from origin data and save
            1. each_chr_sum:  sum_contact for each chr, shape (cells, chrs) eg.(626,23)
            2. each_chr_nonzero_sum:  sum_nonzero for each chr, shape (cells, chrs) eg.(626,23)
            3. each_chr_avg: avg_contact for each chr (each_chr_sum/each_chr_nonzero_sum), shape (cells, chrs) eg.(626,23)
            4. each_cell_sum:  sum_contact for each cell, shape (cells, ) eg.(626,)
            5. each_cell_nonzero_sum:  sum_nonzero for each cell, shape (cells, ) eg.(626,)
            6. each_cell_avg: avg_contact for each cell (each_cell_sum/each_cell_nonzero_sum), shape (cells, ) eg.(626,)
            7. each_chr_sparsity:  sparsity for each chr, sum(nonzero in chr)/(binsize*binsize), shape (cells, chrs) eg.(626,23)
            8. each_cell_sparsity:  sparsity for each cell, sum(nonzero in cell)/sum(binsize*binsize), shape (cells, chrs) eg.(626,)
            9. bulk: bulk contact map for chrs, sum(contact matrix), shape(chrs,) eg.(23,)
        """

        each_chr_nonzero_sum = np.zeros((self.cell_num, self.chr_num), dtype=int)
        each_chr_sum = np.zeros((self.cell_num, self.chr_num), dtype=float)
        bulk = [coo_matrix(([], ([], [])), shape=(chrsize, chrsize)) for chrsize in self.chrsizes]

        bar = tqdm(total=self.cell_num, desc='statistic chr and cell feats',disable=self.no_tqdm)

        # 遍历每个 cell_id
        for cell_id in range(self.cell_num):
            for chr_id in range(self.chr_num):
                cname = self.chrom_list[chr_id]
                m_coo = self.contact_matrices_coo[cname][cell_id]
                each_chr_nonzero_sum[cell_id][chr_id] = m_coo.count_nonzero()
                each_chr_sum[cell_id][chr_id] = m_coo.sum()
                try:
                    bulk[chr_id] += m_coo
                except ValueError:
                    print(bulk[chr_id].shape)
                    print(m_coo.shape)
                    exit(0)
            bar.update(1)

        bar.close()
        
        half_map_size = np.array([(1+chrsize)*chrsize/2 for chrsize in self.chrsizes])

        each_cell_sum = np.sum(each_chr_sum,axis=-1,keepdims=False)
        each_cell_nonzero_sum = np.sum(each_chr_nonzero_sum,axis=-1,keepdims=False)
        each_chr_sparsity = each_chr_nonzero_sum / half_map_size
        each_cell_sparsity = each_cell_nonzero_sum / np.sum(half_map_size)
        each_chr_avg = each_chr_sum / each_chr_nonzero_sum
        each_cell_avg = each_cell_sum / each_cell_nonzero_sum
        bulk = [coo.toarray() for coo in bulk]

        self.statistic_dict = {
            "each_chr_sum": each_chr_sum,
            "each_chr_nonzero_sum": each_chr_nonzero_sum,
            "each_chr_avg": each_chr_avg,
            "each_cell_sum": each_cell_sum,
            "each_cell_nonzero_sum": each_cell_nonzero_sum,
            "each_cell_avg": each_cell_avg,
            "each_chr_sparsity": each_chr_sparsity,
            "each_cell_sparsity": each_cell_sparsity,
            "bulk": bulk
        }


    def get_cm4chr(self,chr_id):
        """
        args:
            chrid: the chr to be get
        get:
            cms4chrid: the chr contact_matrix for all cells (coo_matrix)
        """
        cid = self.chrom_list.index(chr_id)
        chr_size = self.chrsizes[cid]
        cms4chrid = []

        mask1 = self.data[:, 1] == cid
 
        data4chr = self.data[mask1]
        weight4chr = self.weight[mask1]
        
        if not np.all(np.diff(data4chr[:, 0]) >= 0):
            # print(f"Warning: Data for {chr_id} is not sorted by cell_id. Sorting now...")
            sort_indices = np.argsort(data4chr[:, 0])
            data4chr = data4chr[sort_indices]
            weight4chr = weight4chr[sort_indices]

        cell_starts = np.searchsorted(data4chr[:, 0], np.arange(self.cell_num), side='left')
        cell_ends = np.searchsorted(data4chr[:, 0], np.arange(self.cell_num), side='right')


        for cell_id in range(self.cell_num):
            data_celli = data4chr[cell_starts[cell_id]:cell_ends[cell_id]]
            weight_celli = weight4chr[cell_starts[cell_id]:cell_ends[cell_id]]       
            m_coo = coo_matrix((weight_celli, 
                                (data_celli[:, 2] - self.start_pos[cid], data_celli[:,3] - self.start_pos[cid])), 
                                shape=(chr_size,chr_size))
 

            cms4chrid.append(m_coo)
        
        self.contact_matrices_coo[chr_id] = coo_full(np.array(cms4chrid),rm_diag=True)


    def get_cm4chr_all(self):
        bar = trange(self.chr_num,desc='get contact matrix',disable=self.no_tqdm)
        for c in self.chrom_list:
            self.get_cm4chr(c)
            bar.update(1)
        bar.close()


    def raw_cm_impute(self,chr_id,impute_mode='schicluster',pad=1,rp=0.5,with_log=False):
        if impute_mode is None:
            self.imputed_cm_coo[chr_id] = np.copy(self.contact_matrices_coo[chr_id])
        else:
            if self.memory_limit:
                my_cell_adj_all = coo_dense_exchange(self.contact_matrices_coo[chr_id],to_which='dense')
                my_cell_adj_all = schicluster_gpu(my_cell_adj_all,pad,rp,False,self.default_device,with_log,batch_size=600)
                gc.collect()
                torch.cuda.empty_cache()
                print("complete imputation")
                return my_cell_adj_all
            else:
                my_cell_adj_all = np.copy(self.contact_matrices_coo[chr_id])
                my_cell_adj_all = coo_dense_exchange(my_cell_adj_all,to_which='dense')
                my_cell_adj_all = schicluster_gpu(my_cell_adj_all,pad,rp,False,self.default_device,with_log)
                my_cell_adj_all = coo_dense_exchange(my_cell_adj_all,to_which='coo')
                self.imputed_cm_coo[chr_id] = my_cell_adj_all
                return my_cell_adj_all


    def raw_cm_impute_all(self,impute_mode='schicluster',pad=1,rp=0.5):
        bar = trange(self.chr_num, desc=f'{impute_mode if impute_mode is not None else "None"} imputing (pad={pad}, rp={rp})',disable=self.no_tqdm)
        for c in self.chrom_list:
            self.raw_cm_impute(c,impute_mode,pad,rp)
            bar.update(1)
        bar.close()


    def gen_chr_feat(self,chr_id,feat_mode='tsvd',impute_mode='schicluster',pad=1,rp=0.5,with_log=False):
        """
        my_cell_adj_all: a list [chr_matrix for all cells (csr) and each is normalized by contact_sum]
        """
        cid = self.chrom_list.index(chr_id)
        chr_size = self.chrsizes[cid]

        if impute_mode is not None:
            my_cell_adj_all = self.raw_cm_impute(chr_id,impute_mode,pad,rp,with_log)
            gc.collect()
        else:
            my_cell_adj_all = np.copy(self.contact_matrices_coo[chr_id])

        if chr_id not in self.chr_feats_dict:
            self.chr_feats_dict[chr_id] = {}


        if feat_mode == 'BCP':
            cell_feat = SICP_BCP(my_cell_adj_all, chr_size, 'BCP', device=self.default_device)
        
        elif feat_mode == 'SICP':
            cell_feat = SICP_BCP(my_cell_adj_all, chr_size, 'SICP', device=self.default_device)

        else:
            if self.memory_limit:
                dense_cm = my_cell_adj_all
                del my_cell_adj_all
            else:
                dense_cm = coo_dense_exchange(my_cell_adj_all,to_which='dense')
            if rp == -1:
                vector_cm = flatten_ndiags(dense_cm, -1)
            else:
                vector_cm = dense_cm.reshape((dense_cm.shape[0], -1))
            if self.memory_limit:
                del dense_cm
                gc.collect()
            mask = np.array(np.sum(vector_cm > 0, axis=0) > min(10, vector_cm.shape[0] - 2))
            mask = mask.reshape((-1))

            vector_cm = vector_cm[:, mask]
            size = chr_size
            mean_, std_ = np.mean(vector_cm), np.std(vector_cm)
            np.clip(vector_cm, a_min=None, a_max=mean_ + 10 * std_, out=vector_cm)
            if feat_mode == 'tsvd':
                cell_feat = TruncatedSVD(n_components=size, algorithm='randomized', n_iter=2, random_state=None).fit_transform(vector_cm)
            else:
                assert 'no such mode!'

            if cell_feat.shape[-1] < size:
                pad_width = ((0, 0), (0, size - cell_feat.shape[-1]))
                cell_feat = np.pad(cell_feat, pad_width, mode='constant', constant_values=0)
                warnings.warn(f'{chr_id} has poor quality in feature generation, be careful.')
                
        self.chr_feats_dict[chr_id][feat_mode] = cell_feat


    def gen_all_chr_feats(self,feat_mode='BCP',impute_mode='schicluster',pad=1,rp=0.5,with_log=False):
        bar = trange(self.chr_num, desc=f'gen {feat_mode} feats')
        for c in self.chrom_list:
            self.gen_chr_feat(c,feat_mode,impute_mode,pad,rp,with_log)
            bar.update(1)
        bar.close()  

    
    def gen_multi_chr_feats(self,multi_feat_mode=['BCP'],impute_mode=None,pad=1,rp=0.5,with_log=False):
        for feat_mode in multi_feat_mode:
            self.gen_all_chr_feats(feat_mode,impute_mode,pad,rp,with_log)
            
    
    def filter_cells_by_criteria(self, 
                             keep_cell_types = ['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Pvalb', 'Sst', 'Vip', 'L23'], 
                             batchid = ['batch_181218', 'batch_190305', 'batch_190315']):
        if keep_cell_types is None or batchid is None:
            return
        
        mask_type = np.isin(self.labels, keep_cell_types)
  
        if self.batchinfo is not None:
            mask_batch = np.isin(self.batchinfo, batchid)
            final_mask = mask_type & mask_batch
        else:
            final_mask = mask_type 
            
        keep_idx = np.flatnonzero(final_mask)
        self.labels = self.labels[keep_idx]
        
        if self.batchinfo is not None:
            self.batchinfo = self.batchinfo[keep_idx]
        
        self.cell_num = len(keep_idx)
        
        for chrom in self.chrom_list:
            self.contact_matrices_coo[chrom] = self.contact_matrices_coo[chrom][keep_idx]
        
        return keep_idx

if __name__ == '__main__':
    pass


