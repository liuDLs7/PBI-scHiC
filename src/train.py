from collections import OrderedDict
import re
import torch
from utils import *
from Process import *
from Modules import *
import multiprocessing
import torch.optim as optim
from tqdm import trange
import pickle
import time
import itertools

class Train:
    def __init__(self,config,mode='embed',no_tqdm=False):
        self.state = mode
        self.config = config

        self.dataset = self.config['dataset']
        self.origin_dir = self.config['data_dir']
        self.model_dir = os.path.join(self.config['model_dir'],self.dataset)
        os.makedirs(self.model_dir,exist_ok=True)

        if mode == 'impute':
            self.impute_init()
        elif mode == 'embed':
            self.embed_init()
        else:
            raise ValueError(f'Unknown mode:{mode}')

        self.no_tqdm = no_tqdm
        self.cpu_num = multiprocessing.cpu_count()
        self.available_devices = get_available_gpus()
        self.gpu_id = self.available_devices[0]
        self.default_device = torch.device(f'cuda:{self.gpu_id}')

    def embed_init(self):
        self.train_chrom_list = self.config['train_chrom_list']
        self.batch_size = self.config['batch_size']
        self.epoch = self.config['epoch']
        self.resolution = int(self.config['resolution'])        
        self.threshold = int(self.config['threshold'])
        self.diags = int(self.threshold // self.resolution) if self.threshold != -1 else -1
        self.dimensions = self.config['embed_dim']
        self.each_epoch = 50
        self.show = False
        self.loss_func = loss_func
        self.memory_limit = False

    def prep_data(self):
        # step1: prep datas
        if self.state == 'impute':
            print('Preparing datas for imputation...')
        else:
            print('Preparing datas for embed...')
        self.processor = Processor(self.config,no_tqdm=self.no_tqdm)
        self.chrom_sizes = self.processor.chrom_sizes
        self.chr_num = len(self.train_chrom_list)
        if "keep_cell_types" in self.config:
            self.keep_idx = self.processor.filter_cells_by_criteria(self.config["keep_cell_types"], ['batch_190305', 'batch_190315'])
            self.batch_size = self.processor.cell_num // 4 + 1
        else:
            self.keep_idx = self.processor.filter_cells_by_criteria(None, None)
        self.cell_num = self.processor.cell_num
        self.labels = self.processor.labels
        self.batchinfo = self.processor.batchinfo
        
        print(f'cell_num:{self.cell_num}')
        if self.state == 'embed':
            self.head_len = []
            self.tail_len = []
            for chrom in self.train_chrom_list:
                head_len, tail_len = get_head_len(self.chrom_sizes[chrom], self.diags)
                self.head_len.append(head_len)
                self.tail_len.append(tail_len)
            print(self.train_chrom_list)
            print(self.head_len)

        ##########################gen cell init feats####################################
        os.makedirs(self.config['feat_dir'], exist_ok=True)
        chr_feats_path = f'{self.config["feat_dir"]}/{self.dataset}.pkl'

        if os.path.exists(chr_feats_path):
            with open(chr_feats_path, "rb") as file:
                self.cell_feat_dict = pickle.load(file)
            print(f'load feats from {chr_feats_path}, including {list(list(self.cell_feat_dict.values())[0].keys())}')

        else:
            print('start gen chr feats')
            self.processor.gen_multi_chr_feats(self.config['extra_feats'])
            self.processor.gen_multi_chr_feats(self.config['main_feat'],'schicluster',1,0.5)
            self.cell_feat_dict = self.processor.chr_feats_dict
            with open(chr_feats_path, "wb") as file:
                pickle.dump(self.cell_feat_dict, file)
            print(f'save feats in {chr_feats_path}, including {list(list(self.cell_feat_dict.values())[0].keys())}')
            print('features have been generated, please rerun')
            exit(0)
            
        if self.keep_idx is not None:
            self.processor.gen_multi_chr_feats(self.config['main_feat'],'schicluster',1,0.5)
            for keys in self.cell_feat_dict.keys():
                self.cell_feat_dict[keys][self.config['main_feat'][0]] = self.processor.chr_feats_dict[keys][self.config['main_feat'][0]]
            
        all_feats_list = []

        self.main_feat = []
        for chrom in self.train_chrom_list:
            self.main_feat.append(self.cell_feat_dict[chrom][self.config['main_feat'][0]])

        self.main_feat_merge = np.concatenate(self.main_feat, axis=-1)
        self.main_feat_merge = StandardScaler().fit_transform(self.main_feat_merge.reshape((-1, 1))).reshape(
                        (len(self.main_feat_merge), -1))
        
        all_feats_list.append(self.main_feat_merge)

        for extra_feat in self.config['extra_feats']:
            extra_feat_list = []
            for chrom in self.train_chrom_list:
                extra_feat_list.append(self.cell_feat_dict[chrom][extra_feat])
            extra_feat_merge = np.concatenate(extra_feat_list, axis=-1)
            if self.keep_idx is not None:
                extra_feat_merge = extra_feat_merge[self.keep_idx,:]
            all_feats_list.append(extra_feat_merge)
        
        self.feats_num = len(all_feats_list)
        if self.feats_num > 1:
            self.cell_feat_origin = np.stack(all_feats_list, axis=1)
        else:
            self.cell_feat_origin = self.main_feat_merge
 
        print('cell_feat_shape:',self.cell_feat_origin.shape)
        self.cell_init = torch.from_numpy(self.cell_feat_origin).float().to(self.default_device)

        ################################################################################

        #######################gen each chr init and target#############################
        if self.state == 'impute':
            self.contact_matrices_coo = []
            for chrom in self.train_chrom_list:
                self.contact_matrices_coo.append(self.processor.contact_matrices_coo[chrom])
            return
        self.chr_target_dict_head = []
        self.chr_target_dict_tail = []
        self.bulk_chr_feat_diag_head = []
        self.bulk_chr_feat_diag_tail = []

        for i, chrom in enumerate(tqdm(self.train_chrom_list, desc='Spliting head and tail', leave=False, disable=self.no_tqdm)):
            coo = self.processor.contact_matrices_coo[chrom]  
            dense = coo_dense_exchange(coo, 'dense')     
            diag  = flatten_ndiags(dense, -1)              
            bulk_diag = np.sum(diag, axis=0, keepdims=True)
            diag = normalize_([diag], self.config)[0]
            head = diag[:, :self.head_len[i]]
            tail = diag[:, self.head_len[i]:]

            self.chr_target_dict_head.append(head)
            self.chr_target_dict_tail.append(tail)
            self.bulk_chr_feat_diag_head.append(head.mean(dim=0, keepdim=True).float().to(self.default_device))
            self.bulk_chr_feat_diag_tail.append(tail.mean(dim=0, keepdim=True).float().to(self.default_device))
            
            del dense, diag, head, tail, bulk_diag, coo
            gc.collect()
            torch.cuda.empty_cache()
    
    def prep_model(self, train_all=False):
        print('Preparing Models for embed...')
        shape_lists = []
        shape_lists.append([self.cell_feat_origin.shape[-1], self.dimensions])
        for ele in self.bulk_chr_feat_diag_head:
            shape_lists.append([ele.shape[-1],self.dimensions])
        self.head_model = ConditionalHiCGenerator(
            shape_lists=shape_lists,
            d_model=self.dimensions,
            nhead=8,
            feats_num=self.feats_num,
            train_all=train_all).float()
        
        self.optimizer_head = torch.optim.Adam(self.head_model.parameters(), lr=5e-3) 
        
        if self.diags != -1:
            shape_lists = []
            shape_lists.append([self.cell_feat_origin.shape[-1], self.dimensions])
            for ele in self.bulk_chr_feat_diag_tail:
                shape_lists.append([ele.shape[-1],self.dimensions//2,self.dimensions])
            self.tail_model = ConditionalHiCGenerator(
                shape_lists=shape_lists,
                d_model=self.dimensions,
                nhead=8,
                feats_num=self.feats_num,
                train_all=train_all).float()

            self.tail_model.cell_encoder = self.head_model.cell_encoder
            
            if train_all:
                print('share mha and mixer')
                self.tail_model.mha = self.head_model.mha
                self.tail_model.mixer = self.head_model.mixer
                self.tail_model.head_tail_fuser = self.head_model.head_tail_fuser
                self.tail_model.head_tail_splitter = self.head_model.head_tail_splitter
                shared_modules = [self.head_model.cell_encoder, self.head_model.mha, self.head_model.mixer, self.head_model.head_tail_fuser, self.head_model.head_tail_splitter]
            else:
                shared_modules = [self.head_model.cell_encoder]
                
            shared_params = list(itertools.chain.from_iterable(m.parameters() for m in shared_modules))
            shared_ids = {id(p) for p in shared_params}

            head_only_params = [p for p in self.head_model.parameters() if id(p) not in shared_ids]
            tail_only_params = [p for p in self.tail_model.parameters() if id(p) not in shared_ids]

            self.optimizer_all = torch.optim.Adam(
                [
                    {"params": shared_params,    "lr": 1e-3},  
                    {"params": head_only_params, "lr": 1e-3},
                    {"params": tail_only_params, "lr": 1e-3},
                ]
            )
            
            self.optimizer_tail = torch.optim.Adam(self.tail_model.parameters(), lr=2e-3) 
            print('cell encoder:', type(self.head_model.cell_encoder))
            
    
    def train_one_batch(self, cell_ids, chr_id, return_recon=False, return_embed=False, use_bulk=False, mode='head', return_sum=False):
        if mode == 'head':
            model = self.head_model
            chr_target_dict = self.chr_target_dict_head
            bulk_chr_diag = self.bulk_chr_feat_diag_head
        else:
            model = self.tail_model
            chr_target_dict = self.chr_target_dict_tail
            bulk_chr_diag = self.bulk_chr_feat_diag_tail

        target_dict = chr_target_dict[chr_id]
        target = target_dict[cell_ids, :]
        
        if self.memory_limit:
            target = target.to(self.default_device, non_blocking=True)

        cell_init_ = self.cell_init[cell_ids, :]
        if use_bulk:
            chr_init = bulk_chr_diag[chr_id]
        else:
            chr_init = torch.mean(target,dim=0,keepdim=True)

        if return_embed:
            return model(cell_init_, chr_init, chr_id, return_embed)
        
        chr_recon = model(cell_init_, chr_init, chr_id)

        if return_recon:
            return chr_recon
        
        loss = self.loss_func(chr_recon.float(), target.float()) 

        if return_sum:
            return loss, torch.sum(target)
        
        return loss


    def train_epochs(self,optimizer,epochs,mode=None):
        rand_cell_ids = list(range(self.cell_num))
        for i in range(epochs):
            bar = trange(self.each_epoch,desc='(Training -)',leave=True,disable=self.no_tqdm)
            epoch_loss = []
            start_time = time.time()
            
            for _ in range(self.each_epoch):
                random.shuffle(rand_cell_ids)
                batch_cell_id = []
                start = 0
                while start < self.cell_num:
                    end = start + self.batch_size
                    if end > self.cell_num:
                        end = self.cell_num
                    if end-start < 10:
                        break
                    batch_cell_id.append(rand_cell_ids[start:end])
                    start = end

                total_loss = []
                for batch in batch_cell_id:
                    optimizer.zero_grad()

                    losses = []
                    for k in range(self.chr_num): 
                        loss_k = self.train_one_batch(cell_ids=batch, chr_id=k, mode=mode)
                        losses.append(loss_k)
                        loss_k.backward()
                        
                    loss = torch.mean(torch.stack(losses))
                    total_loss.append(loss.item())
                    epoch_loss.append(loss.item())
                    
                    optimizer.step() 

                bar.set_description(desc=f'(Training - Loss:{np.mean(total_loss):.3f})')
                bar.update(1)

            bar.close()
                    
    def joint_train(self,epochs=30,lr=1e-3):
        if self.threshold == -1:
            return 
        print('Start joint training...')
        if not self.memory_limit:
            self.chr_target_dict_head = [x.to(self.default_device) for x in self.chr_target_dict_head]
        self.head_model.to(self.default_device, non_blocking=True)
 
        if self.diags != -1:
            if not self.memory_limit:
                self.chr_target_dict_tail = [x.to(self.default_device) for x in self.chr_target_dict_tail]
            self.tail_model.to(self.default_device, non_blocking=True)

        optimizer = self.optimizer_all
        set_learning_rate(optimizer, lr)
        rand_cell_ids = list(range(self.cell_num))
        for i in range(epochs):
            bar = trange(self.each_epoch,desc='(Training -)',leave=True,disable=self.no_tqdm)
            epoch_loss = []
            start_time = time.time()
            
            for _ in range(self.each_epoch):
                random.shuffle(rand_cell_ids)
                batch_cell_id = []
                start = 0
                while start < self.cell_num:
                    end = start + self.batch_size
                    if end > self.cell_num:
                        end = self.cell_num
                    if end-start < 10:
                        break
                    batch_cell_id.append(rand_cell_ids[start:end])
                    start = end

                total_loss = []
                rrs = []
                for batch in batch_cell_id:
                    optimizer.zero_grad()

                    losses = []
                    rr = []
                    for k in range(self.chr_num):
                        hloss_k, hsum_k = self.train_one_batch(cell_ids=batch, chr_id=k, mode='head', return_sum=True)
                        tloss_k, tsum_k = self.train_one_batch(cell_ids=batch, chr_id=k, mode='tail', return_sum=True)
                        ht_ratio = (hsum_k / self.head_len[k]) / (tsum_k / self.tail_len[k] + 1e-8)
                        ttloss_k = hloss_k / ht_ratio + tloss_k
                        ttloss_k.backward()
                        losses.append(ttloss_k)
                        rr.append(ht_ratio)
                    
                    rrs.append(torch.mean(torch.stack(rr)).cpu().numpy())
                    loss = torch.mean(torch.stack(losses))
                    total_loss.append(loss.item())
                    epoch_loss.append(loss.item())

                    optimizer.step() 

                bar.set_description(desc=f'(Training - Loss:{np.mean(total_loss):.3f}), HT_ratio:{np.mean(rrs):.3f}')
                bar.update(1)

            bar.close()
            
    def train(self):
        print('Start training...')
        if self.show:
            from visualization import DynamicUMAPVisualizer
            self.visualizer = DynamicUMAPVisualizer(title=self.dataset)
            cell_feats = self.fit()
            self.visualizer.update(cell_feats, self.labels, f'{self.dataset}')

        if not self.memory_limit:
            self.chr_target_dict_head = [x.to(self.default_device) for x in self.chr_target_dict_head]
        self.head_model.to(self.default_device, non_blocking=True)
            
        self.train_epochs(self.optimizer_head, self.epoch, mode='head')
        if not self.memory_limit:
            self.chr_target_dict_head = [x.cpu() for x in self.chr_target_dict_head]
        gc.collect()
        torch.cuda.empty_cache()

        if self.diags != -1:
            if not self.memory_limit:
                self.chr_target_dict_tail = [x.to(self.default_device) for x in self.chr_target_dict_tail]
            self.tail_model.to(self.default_device, non_blocking=True)
            self.train_epochs(self.optimizer_tail, self.epoch, mode='tail')
            if not self.memory_limit:
                self.chr_target_dict_tail = [x.cpu() for x in self.chr_target_dict_tail]
        else:
            if not self.memory_limit:
                self.chr_target_dict_head = [x.to(self.default_device) for x in self.chr_target_dict_head]
            self.head_model.to(self.default_device, non_blocking=True)
            self.train_epochs(self.optimizer_head, self.epoch, mode='head')
            if not self.memory_limit:
                self.chr_target_dict_head = [x.cpu() for x in self.chr_target_dict_head]
        gc.collect()
        torch.cuda.empty_cache()
    
    def train_one_batch_all(self, cell_ids, chr_id=None, return_recon=False, return_embed=False, use_bulk=False, mode='head'):
        if mode == 'head':
            model = self.head_model
            chr_target_dict = self.chr_target_dict_head
            bulk_chr_diag = self.bulk_chr_feat_diag_head
        else:
            model = self.tail_model
            chr_target_dict = self.chr_target_dict_tail
            bulk_chr_diag = self.bulk_chr_feat_diag_tail

        targets = []
        for i in range(self.chr_num):
            target_dict = chr_target_dict[i]
            target = target_dict[cell_ids, :]
            targets.append(target)

        cell_init_ = self.cell_init[cell_ids, :]
        if use_bulk:
            chr_init = bulk_chr_diag
        else:
            chr_inits = []
            for k in range(self.chr_num):
                target = targets[k]
                chr_inits.append(torch.mean(target,dim=0,keepdim=True))

        if return_embed:
            return model(cell_init_, chr_inits, chr_id, return_embed, train_all=True)
        
        chr_recons = model(cell_init_, chr_inits, chr_id, train_all=True)

        if return_recon:
            return chr_recons
        
        loss = []
        for k in range(self.chr_num):
            chr_recon = chr_recons[k]
            target = targets[k]
            loss.append(loss_func(chr_recon.float(), target.float(), temperature=1.0,target_power=1.0))
            
        return loss
    
    def train_one_batch_head_tail(self, cell_ids, chr_id=None, return_recon=False, return_embed=False, use_bulk=False, mode=None):
        head_model = self.head_model
        chr_target_dict_head = self.chr_target_dict_head
        bulk_chr_diag_head = self.bulk_chr_feat_diag_head
        
        tail_model = self.tail_model
        chr_target_dict_tail = self.chr_target_dict_tail
        bulk_chr_diag_tail = self.bulk_chr_feat_diag_tail
        
        targets_head = []
        targets_tail = []
        for i in range(self.chr_num):
            target_dict_head = chr_target_dict_head[i]
            target_head = target_dict_head[cell_ids, :]
            targets_head.append(target_head)
            
            target_dict_tail = chr_target_dict_tail[i]
            target_tail = target_dict_tail[cell_ids, :]
            targets_tail.append(target_tail)
            
        cell_init_ = self.cell_init[cell_ids, :]
        if use_bulk:
            chr_inits_head = bulk_chr_diag_head
            chr_inits_tail = bulk_chr_diag_tail
        else:
            chr_inits_head = []
            chr_inits_tail = []
            for k in range(self.chr_num):
                target_head = targets_head[k]
                chr_inits_head.append(torch.mean(target_head,dim=0,keepdim=True))
                
                target_tail = targets_tail[k]
                chr_inits_tail.append(torch.mean(target_tail,dim=0,keepdim=True))
        
        cell_embed, chr_embed_head = head_model.encoder(cell_init_, chr_inits_head, chr_id, extra_feats=None, train_all=True)
        _, chr_embed_tail = tail_model.encoder(cell_init_, chr_inits_tail, chr_id, extra_feats=None, train_all=True)
        embed = head_model.ccfusion(cell_embed, [chr_embed_head, chr_embed_tail], train_all=True)
        
        if return_embed:
            return embed

        chr_recons_head = head_model.decoder(embed[0], chr_id, train_all=True)
        chr_recons_tail = tail_model.decoder(embed[1], chr_id, train_all=True)
        
        if return_recon:
            return chr_recons_head, chr_recons_tail
        
        loss = []
        for k in range(self.chr_num):
            chr_recon_head = chr_recons_head[k]
            target_head = targets_head[k]
            loss.append(loss_func(chr_recon_head.float(), target_head.float(), temperature=1.0,target_power=1.0))
            
            chr_recon_tail = chr_recons_tail[k]
            target_tail = targets_tail[k]
            loss.append(loss_func(chr_recon_tail.float(), target_tail.float(), temperature=1.0,target_power=1.0))
        
        return loss

    def train_epochs_all(self,optimizer,epochs,mode=None):
        rand_cell_ids = list(range(self.cell_num))
        for i in range(epochs):
            bar = trange(self.each_epoch,desc='(Training -)',leave=False,disable=self.no_tqdm)
            start_time = time.time()
            
            for _ in range(self.each_epoch):
                random.shuffle(rand_cell_ids)
                batch_cell_id = []
                start = 0
                while start < self.cell_num:
                    end = start + self.batch_size
                    if end > self.cell_num:
                        end = self.cell_num
                    if end-start < 10:
                        break
                    batch_cell_id.append(rand_cell_ids[start:end])
                    start = end

                optimizer.zero_grad()
                loss = []
                for batch in batch_cell_id:
                    loss += self.train_one_batch_head_tail(cell_ids=batch, mode='head_tail', use_bulk=False)
                loss = torch.mean(torch.stack(loss))
                loss.backward()
                optimizer.step() 

                bar.set_description(desc=f'(Training - Loss:{loss.item():.3f})')
                bar.update(1)

            bar.close()

    def train_all(self):
        print('Start trianing all...')
        if self.show:
            from visualization import DynamicUMAPVisualizer
            self.visualizer = DynamicUMAPVisualizer(title=self.dataset)
            cell_feats = self.fit()
            self.visualizer.update(cell_feats, self.labels, f'{self.dataset}')

        self.chr_target_dict_head = [x.to(self.default_device) for x in self.chr_target_dict_head]
        self.chr_target_dict_tail = [x.to(self.default_device) for x in self.chr_target_dict_tail]
        self.train_epochs_all(self.optimizer_all, self.epoch, mode='all')
        self.chr_target_dict_head = [x.cpu() for x in self.chr_target_dict_head]
        self.chr_target_dict_tail = [x.cpu() for x in self.chr_target_dict_tail]
        gc.collect()
        torch.cuda.empty_cache()
        
    def save_model(self):
        save_dir = self.model_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving head model to {os.path.join(save_dir, 'head_model.pth')}")
        torch.save(self.head_model.state_dict(), os.path.join(save_dir, 'head_model.pth'))
        if self.diags != -1:
            print(f"Saving tail model to {os.path.join(save_dir, 'tail_model.pth')}")
            torch.save(self.tail_model.state_dict(), os.path.join(save_dir, 'tail_model.pth'))

    def load_model(self):
        load_dir = self.model_dir
        model_to_load = self.head_model
        model_path = os.path.join(load_dir, 'head_model.pth')
        model_name = 'Head'
        print(f"Loading {model_name} model from '{model_path}'...")
        state_dict = torch.load(model_path, map_location=self.default_device)
        model_to_load.load_state_dict(state_dict)
        print(f"{model_name} model loaded successfully.")

        model_to_load = self.tail_model
        model_path = os.path.join(load_dir, 'tail_model.pth')
        model_name = 'Tail'
        print(f"Loading {model_name} model from '{model_path}'...")
        state_dict = torch.load(model_path, map_location=self.default_device)
        model_to_load.load_state_dict(state_dict)
        print(f"{model_name} model loaded successfully.") 
            
    def fit(self):  
        model = self.head_model
        trained_cell_feats = model.fit_predict(self.cell_init.clone().cpu().detach().numpy())
        return trained_cell_feats

    def fit_chrs(self):
        head_encoded = []
        tail_encoded = []
        
        self.head_model.eval()
        self.tail_model.eval()
        cell_ids = list(range(self.cell_num))

        with torch.no_grad():
            for i in range(self.chr_num):
                self.chr_target_dict_head = [x.to(self.default_device) for x in self.chr_target_dict_head]
                head_encoded.append(self.train_one_batch(cell_ids=cell_ids, chr_id=i, 
                                                        return_embed=True, mode='head').cpu().detach().numpy())
                self.chr_target_dict_head = [x.cpu() for x in self.chr_target_dict_head]
                
                self.chr_target_dict_tail = [x.to(self.default_device) for x in self.chr_target_dict_tail]
                tail_encoded.append(self.train_one_batch(cell_ids=cell_ids, chr_id=i, 
                                                        return_embed=True, mode='tail').cpu().detach().numpy())
                self.chr_target_dict_tail = [x.cpu() for x in self.chr_target_dict_tail]

        self.head_model.train()
        self.tail_model.train()   
        
        return np.array(head_encoded), np.array(tail_encoded) 
    
    def save_model_n_times(self):
        save_dir = self.model_dir
        save_dir = os.path.join(save_dir, 'n_times_models')
        os.makedirs(save_dir, exist_ok=True)
        for i in range(1,101):
            model_path = os.path.join(save_dir, f'head_model_{i}.pth')
            if os.path.exists(model_path):
                continue
            print(f"Saving head model to {model_path}")
            torch.save(self.head_model.state_dict(), model_path)
            break

        for i in range(1,101):
            model_path = os.path.join(save_dir, f'tail_model_{i}.pth')
            if os.path.exists(model_path):
                continue
            print(f"Saving tail model to {model_path}")
            torch.save(self.tail_model.state_dict(), model_path)
            break

    def load_model_n_times(self, i):
        load_dir = self.model_dir
        load_dir = os.path.join(load_dir, 'n_times_models')
    
        model_path = os.path.join(load_dir, f'head_model_{i}.pth')
        model_to_load = self.head_model
        model_name = 'Head'
        print(f"Loading {model_name} model from '{model_path}'...")
        state_dict = torch.load(model_path, map_location=self.default_device)
        model_to_load.load_state_dict(state_dict)
        print(f"{model_name} model loaded successfully.")

        model_path = os.path.join(load_dir, f'tail_model_{i}.pth')
        model_to_load = self.tail_model
        model_name = 'Tail'
        print(f"Loading {model_name} model from '{model_path}'...")
        state_dict = torch.load(model_path, map_location=self.default_device)
        model_to_load.load_state_dict(state_dict)
        print(f"{model_name} model loaded successfully.")

    # IMPUTE PART
    def impute_init(self):
        self.impute_batch = 32
        print("imputation batch size:", self.impute_batch)
        self.impute_epoch = 20
        self.resolution = self.config['impute_res']
        self.dimensions = self.config['embed_dim']
        self.bound = self.config['bound']
        self.train_chrom_list = self.config['impute_chrom_list']
        self.each_epoch = 50
        self.diags = -1

    def prep_data_impute(self):
        self.part = self.bound is not None
        self.prep_data()
        contact_matrices_coo = self.contact_matrices_coo

        if self.bound is None:
            target_matrices_block = [coo_dense_exchange(x,'dense') for x in contact_matrices_coo]
        else:
            start = int(self.bound[0] // self.resolution)
            end = int(self.bound[1] // self.resolution)
            target_matrices_block = [slice_coo_batch(x, start, end) for x in contact_matrices_coo]

        self.block_sizes = [x.shape[-1] for x in target_matrices_block]
        self.raw_matrices_numpy = [x for x in target_matrices_block]
        self.raw_matrices_diag = [flatten_ndiags(x,-1) for x in self.raw_matrices_numpy]
        self.raw_matrices_diag = normalize_(self.raw_matrices_diag)
        self.target_matrices = [torch.clamp(x, min=1e-8, max=5).float().to(self.default_device) for x in self.raw_matrices_diag]
        self.bulk_matrices = [torch.mean(x,dim=0,keepdim=True).float().to(self.default_device) for x in self.target_matrices]

    def prep_model_impute(self):
        print('Preparing models for imputation...')
        try:
            cell_encoder = MultiFeatMix(self.cell_init.shape[-1], self.feats_num, self.dimensions)
            self.load_part(cell_encoder, 'cell_encoder')
        except:
            try:
                cell_encoder = FeedForward([self.cell_init.shape[-1], self.dimensions])
                self.load_part(cell_encoder, 'cell_encoder')
            except:
                raise ValueError('No pretrained cell encoder found for imputation!')

        shape_lists = []
        shape_lists.append([2,1])

        for ele in self.target_matrices:
            shape_lists.append([ele.shape[-1],self.dimensions])

        for param in cell_encoder.parameters():
            param.requires_grad = True
        self.impute_model = ConditionalHiCGenerator(shape_lists=shape_lists,
                                                    d_model=self.dimensions,
                                                    nhead=8)
        
        self.impute_model.cell_encoder = cell_encoder
        self.impute_model = self.impute_model.float().to(self.default_device, non_blocking=True)
        self.impute_optimizer = optim.Adam(self.impute_model.parameters(), lr=1e-3)

    def train_one_batch_impute(self, cell_ids, chr_id, return_recon=False, return_embed=False, use_bulk=False):
        model = self.impute_model
        target_dict = self.target_matrices[chr_id]
        target = target_dict[cell_ids, :]

        cell_init_ = self.cell_init[cell_ids, :]
        if use_bulk:
            chr_init = self.bulk_matrices[chr_id]
        else:
            chr_init = torch.mean(target,dim=0,keepdim=True)

        if return_embed:
            return model(cell_init_, chr_init, chr_id, return_embed=True)
        
        chr_recon = model(cell_init_, chr_init, chr_id)

        chr_recon = F.softplus(chr_recon)
        chr_recon = torch.clamp(chr_recon, min=1e-8, max=10)

        if return_recon:
            return chr_recon
        
        loss = loss_func(chr_recon, target, temperature=1.0,target_power=1.0)

        return loss

    def train_impute(self):
        print('Start training imputation...')
        optimizer = self.impute_optimizer
        rand_cell_ids = list(range(self.cell_num))
        for i in range(self.impute_epoch):
            bar = trange(self.each_epoch,desc='(Inputation Training -)',leave=False,disable=self.no_tqdm)
            epoch_loss = []
            start_time = time.time()
            
            for _ in range(self.each_epoch):
                random.shuffle(rand_cell_ids)
                batch_cell_id = []
                start = 0
                while start < self.cell_num:
                    end = start + self.impute_batch
                    if end > self.cell_num:
                        end = self.cell_num
                    # if end-start < 10:
                    #     break
                    batch_cell_id.append(rand_cell_ids[start:end])
                    start = end

                total_loss = []
                for batch in batch_cell_id:
                    optimizer.zero_grad()

                    losses = []
                    for k in range(len(self.target_matrices)):
                        loss_k = self.train_one_batch_impute(cell_ids=batch, chr_id=k)
                        losses.append(loss_k)
                        loss_k.backward()

                    loss = torch.mean(torch.stack(losses))
                    total_loss.append(loss.item())
                    epoch_loss.append(loss.item())

                    # loss.backward() 
                    optimizer.step() 

                bar.set_description(desc=f'(Inputation Training - Loss:{np.mean(total_loss):.3f})')
                bar.update(1)

            bar.close()

            print(f"[Epoch {i+1}/{self.impute_epoch}]",end='\t')
            print(f'Avg loss:{np.mean(epoch_loss):.3f} | '
                f'use time:{time.time()-start_time:.2f} seconds')


    def save_model_impute(self):
        part = 'part' if self.part else ''
        res_str = format_number(self.resolution)
        save_dir = self.model_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'inputation_model_{res_str}{part}.pth')
        print(f"Saving inputation model to {save_path}")
        torch.save(self.impute_model.state_dict(), save_path)
    
    def load_model_impute(self):
        part = 'part' if self.part else ''
        res_str = format_number(self.resolution)
        load_dir = self.model_dir
        model_to_load = self.impute_model
        model_path = os.path.join(load_dir, f'inputation_model_{res_str}{part}.pth')
        model_name = 'Imputation'
        print(f"Loading {model_name} model from '{model_path}'...")
        state_dict = torch.load(model_path, map_location=self.default_device)
        model_to_load.load_state_dict(state_dict)
        print(f"{model_name} model loaded successfully.")

    def load_part(self, model_to_load, part='cell_encoder'):
        load_dir = self.model_dir
        model_path = os.path.join(load_dir, 'head_model.pth')
        print(f"Loading {part} model from '{model_path}'...")
        full_state_dict = torch.load(model_path, map_location=self.default_device)
        part_state_dict = OrderedDict()
        prefix = part + '.'
        for key, value in full_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                part_state_dict[new_key] = value
        
        if not part_state_dict:
            available_keys = list(full_state_dict.keys())
            raise KeyError(f"No keys with prefix '{prefix}' found in the checkpoint. "
                        f"First 5 available keys: {available_keys[:5]}")

        model_to_load.load_state_dict(part_state_dict, strict=True)
        print(f"'{part}' model loaded successfully with {len(part_state_dict)} tensors.")

    def fit_impute(self, pad=0, row_stride=1, col_stride=1):
        if self.target_matrices[0].dim() == 3:
            res = self.fit_impute_2d()
        else:
            res = self.fit_impute_1d()
        
        if pad > 0:
            res = [neighbor_ave_gpu(x, pad=1, return_Tensor=False, device=self.default_device, row_stride=row_stride, col_stride=col_stride) for x in res]

        return res

    def fit_impute_2d(self):
        imputed_matrices = []
        cell_ids = list(range(self.cell_num))
        batch_cell_id = []
        start = 0
        while start < self.cell_num:
            end = start + 1
            if end > self.cell_num:
                end = self.cell_num
            batch_cell_id.append(cell_ids[start:end])
            start = end

        self.impute_model.eval()
        with torch.no_grad():
            for i in range(len(self.target_matrices)):
                one_chr_imputed = []
                for batch in batch_cell_id:
                    batch_imputed = self.train_one_batch_impute(cell_ids=batch, chr_id=i, return_recon=True)
                    one_chr_imputed.append(batch_imputed.detach().cpu().numpy())
                one_chr_imputed = np.concatenate(one_chr_imputed, axis=0)
                imputed_matrices.append(one_chr_imputed)

        self.impute_model.train()

        return imputed_matrices

    def fit_impute_1d(self):
        imputed_matrices_diag = []
        cell_ids = list(range(self.cell_num))
        batch_cell_id = []
        start = 0
        while start < self.cell_num:
            end = start + 1
            if end > self.cell_num:
                end = self.cell_num
            batch_cell_id.append(cell_ids[start:end])
            start = end

        self.impute_model.eval()
        with torch.no_grad():
            for i in range(len(self.target_matrices)):
                one_chr_imputed = []
                for batch in batch_cell_id:
                    batch_imputed = self.train_one_batch_impute(cell_ids=batch, chr_id=i, return_recon=True, use_bulk=False)
                    one_chr_imputed.append(batch_imputed.detach().cpu().numpy())
                one_chr_imputed = np.concatenate(one_chr_imputed, axis=0)
                imputed_matrices_diag.append(one_chr_imputed)

        self.impute_model.train()

        imputed_matrices = [dense_full(unflatten_all_ndiags(x, self.block_sizes[i])) for i,x in enumerate(imputed_matrices_diag)]

        return imputed_matrices
        
    
    def fit_impute_cell_feat(self):
        self.impute_model.eval()
        with torch.no_grad():
            cell_embed = self.impute_model.cell_encoder(self.cell_init)
        self.impute_model.train()
        return cell_embed.detach().cpu().numpy()
    
