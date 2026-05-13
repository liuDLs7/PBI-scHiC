import torch.nn as nn
import torch
import numpy as np
from utils import *


class FeedForward(nn.Module):
    def __init__(self, dims, layer_norm=False, dropout=None, activation=None, add_activation=False, use_conv=False, use_bias=True):
        super(FeedForward, self).__init__()
        if layer_norm:
            self.layer_norm = nn.LayerNorm(dims[0])
        else:
            self.layer_norm = None
        
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if activation is not None:
            self.activation = activation
        else:
            self.activation = F.silu

        self.add_activaton = add_activation
        self.use_bias = use_bias

        self.use_conv = use_conv    
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 7, padding='same'),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 32, 7, padding='same'),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.flat = nn.Flatten()

            dims[0] = dims[0] // 2 // 2 * 32

        self.w_stack = []
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1], bias=self.use_bias)

            self.w_stack.append(layer)
        self.w_stack = nn.ModuleList(self.w_stack)

    
    def forward(self, x):
        output = self.layer_norm(x) if self.layer_norm is not None else x

        if self.use_conv:
            output = output.unsqueeze(1)
            output = self.conv(output)
            output = self.flat(output)


        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = self.activation(output)
            output = self.dropout(output) if self.dropout is not None else output

        output = self.w_stack[-1](output)

        if self.add_activaton:
            output = self.activation(output)
            output = self.dropout(output) if self.dropout is not None else output

        return output


class Feat_Mixer(nn.Module):
    def __init__(
            self,
            d_model):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.gate_linear = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        batch = x.shape[0]
        gate = torch.sigmoid(self.gate_linear(x.view(batch, -1)))
        output = gate * x[:,0,:] + (1 - gate) * x[:,1,:]
        return output
    

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=-1)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_k=8,
            d_v=8,
            dropout=None):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        
        self.fc = FeedForward([n_head * d_v, d_model], dropout=dropout)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
            

    def forward(self, _q, _k=None, _v=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        if _k is None or _v is None:
            _k = _q
            _v = _q
        q = self.layer_norm1(_q)
        k = self.layer_norm2(_k)
        v = self.layer_norm3(_v)
        
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape
        
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
  
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  
        
        fusion, attn = self.attention(q, k, v)
        fusion = fusion.view(n_head, sz_b, len_q, d_v)
        fusion = fusion.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) 
            
        fusion = self.fc(fusion)
        
        return fusion, attn


class MultiFeatMix(nn.Module):
    def __init__(self, d_model, num_feats, to_dim):
        super(MultiFeatMix, self).__init__()
        self.d_model = d_model
        self.num_feats = num_feats
        self.ffs = []
        for _ in range(num_feats):
            self.ffs.append(FeedForward([d_model, to_dim])) 
        self.ffs = nn.ModuleList(self.ffs)
    
    def forward(self, x):
        outputs = []
        for i in range(self.num_feats):
            processed_feat = self.ffs[i](x[:, i, :])
            outputs.append(processed_feat)

        outputs = torch.stack(outputs, dim=1)

        return torch.sum(outputs, dim=1)


class ConditionalHiCGenerator(nn.Module):
    def __init__(self, shape_lists, d_model, nhead, feats_num=1, extra_feats=None, train_all=False):
        super().__init__()
        self.d_model = d_model
        self.train_all = train_all

        if extra_feats is not None:
            self.use_extra = True
            self.extra_encoder = FeedForward([extra_feats.shape[-1], d_model], layer_norm=False,
                                             dropout=None, add_activation=False)
        
        else:
            self.use_extra = False

        self.feats_num = feats_num
        if feats_num > 1:
            self.cell_encoder = MultiFeatMix(shape_lists[0][0], feats_num, d_model)
        
        else:
            self.cell_encoder = FeedForward(shape_lists[0]) 

        self.chr_encoders = []
        for i in range(1,len(shape_lists)):
             self.chr_encoders.append(FeedForward(shape_lists[i],add_activation=True)) 
        self.chr_encoders = nn.ModuleList(self.chr_encoders)

        self.chr_decoders = []
        for i in range(1,len(shape_lists)):
            reversed_shape_list = shape_lists[i][::-1]
            self.chr_decoders.append(FeedForward(reversed_shape_list))
        self.chr_decoders = nn.ModuleList(self.chr_decoders)
 
        self.mha = MultiHeadAttention(nhead,d_model)
        self.mixer = Feat_Mixer(d_model)

        self.activation = F.silu
        
        if self.train_all:
            self.all_chr_encoder = FeedForward([(len(shape_lists)-1) * d_model, d_model], add_activation=False)
            self.all_chr_decoder = FeedForward([d_model, (len(shape_lists)-1) * d_model], add_activation=False)
            self.head_tail_fuser = FeedForward([2 * d_model, d_model], add_activation=False)
            self.head_tail_splitter = FeedForward([d_model, 2 * d_model], add_activation=False)
        
        self.flag = True
        
    def cell_encode(self, x, extra_feats=None):
        cell_embed = self.cell_encoder(x)
        if self.use_extra and extra_feats is not None:
            extra_embed = self.extra_encoder(extra_feats)
            cell_embed = cell_embed + extra_embed
        return cell_embed
    
    def chr_encode(self, x, chr_id):
        return self.chr_encoders[chr_id](x)

    def chr_decode(self, x, chr_id):
        return self.chr_decoders[chr_id](x)
    
    def encoder(self, cell_feat, chr_feat, chr_id=None, extra_feats=None, train_all=False):
        self.batchsize = cell_feat.shape[0]
        cell_embed = self.cell_encode(cell_feat, extra_feats)
        
        if train_all:
            all_chr_embed = []
            for i in range(len(self.chr_encoders)):
                chr_embed = self.chr_encode(chr_feat[i], i)
                all_chr_embed.append(chr_embed)
            all_chr_embed = torch.cat(all_chr_embed, dim=1)
            chr_embed = self.all_chr_encoder(all_chr_embed)
        else:
            chr_embed = self.chr_encode(chr_feat, chr_id)
            
        return cell_embed, chr_embed
    
    def decoder(self, embed, chr_id=None, train_all=False):
        if train_all:
            decoded_all = self.all_chr_decoder(embed)
            reconstructed = []
            for i in range(len(self.chr_decoders)):
                start_idx = i * self.d_model
                end_idx = (i + 1) * self.d_model
                chr_decoded = self.chr_decode(decoded_all[:, start_idx:end_idx], i)
                reconstructed.append(chr_decoded)
        else:
            reconstructed = self.chr_decode(embed, chr_id)
            
        return reconstructed
    
    
    def ccfusion(self, cell_embed, chr_embed, chr_id=None, train_all=False):
        if train_all and isinstance(chr_embed, list):
            chr_embed = torch.cat(chr_embed, dim=-1)
            chr_embed = self.head_tail_fuser(chr_embed)
        else:
            train_all = False
        if chr_embed.shape[0] == 1 and self.batchsize > 1:
            chr_embed = chr_embed.repeat(self.batchsize, 1)
        cat_embed = torch.cat([cell_embed.unsqueeze(1), chr_embed.unsqueeze(1)], dim=1)
        
        attn_mixed, attn = self.mha(cat_embed, cat_embed, cat_embed)
        embed = self.mixer(attn_mixed)
        if train_all:
            embed = self.head_tail_splitter(embed)
            embed = torch.split(embed, self.d_model, dim=-1)
        return embed
    
    
    def forward(self, cell_feat, chr_feat, chr_id=None, return_embed=False, extra_feats=None, train_all=False):
        cell_embed, chr_embed = self.encoder(cell_feat, chr_feat, chr_id, extra_feats, train_all)
        embed = self.ccfusion(cell_embed, chr_embed, chr_id, train_all)
        if return_embed:
            return embed
        embed = self.activation(embed) # 0130 add
        reconstructed = self.decoder(embed, chr_id, train_all)
            
        return reconstructed

    def fit_predict(self, x:np.ndarray):
        self.eval()
        current_device = next(self.parameters()).device
        x = torch.from_numpy(x).float().to(current_device, non_blocking=True)
        with torch.no_grad():
            encoded = self.cell_encode(x)
        self.train()
        torch.cuda.empty_cache()
        return encoded.cpu().detach().numpy()
    
    def reset_blocks(self, cell_encoder=None, chr_encoders=None, chr_decoders=None):
        if cell_encoder is not None:
            self.cell_encoder = cell_encoder
        if chr_encoders is not None:
            self.chr_encoders = chr_encoders
        if chr_decoders is not None:
            self.chr_decoders =  chr_decoders
            

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    

class FeatureEncoder(nn.Module):
    def __init__(self, input_seq_len, conv_channels, linear_features, 
                 kernel_size=7, dp=None, stride=1, pad=1, activation=nn.ReLU, last=False, pool=True):
        super(FeatureEncoder, self).__init__()

        self.last = last
        self.num_conv_layers = len(conv_channels)
        self.num_linear_layers = len(linear_features)
        pad = kernel_size // 2 if not pool else pad

        if self.num_conv_layers > 0:
            current_seq_len = input_seq_len
            conv_modules = []
            in_c = 1
            for out_c in conv_channels:
                conv_layer = nn.Conv1d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad
                )
                # nn.init.kaiming_normal_(conv_layer.weight, mode="fan_in", nonlinearity="relu")
                conv_modules.append(conv_layer)
                conv_modules.append(nn.BatchNorm1d(out_c))
                conv_modules.append(activation())
                if pool:
                    conv_modules.append(nn.MaxPool1d(kernel_size=2)) 
                if dp is not None:
                    conv_modules.append(nn.Dropout(dp))
                in_c = out_c

                current_seq_len = (current_seq_len + 2 * pad - kernel_size) // stride + 1
                if pool:
                    current_seq_len = current_seq_len // 2
            
            self.conv_net = nn.Sequential(*conv_modules)
            
            linear_input_size = conv_channels[-1] * current_seq_len

        else:
            self.conv_net = nn.Identity()
            linear_input_size = input_seq_len

        if self.num_linear_layers > 0:
            fc_modules = []
            current_feature_size = linear_input_size
            
            for i, out_feature in enumerate(linear_features):
                linear_layer = nn.Linear(in_features=current_feature_size, out_features=out_feature, bias=False)
                # nn.init.kaiming_normal_(linear_layer.weight, mode="fan_in", nonlinearity="relu")
                fc_modules.append(linear_layer)
                if not (i == self.num_linear_layers - 1 and self.last):
                    fc_modules.append(activation())
                    if dp is not None:
                        fc_modules.append(nn.Dropout(dp))
                
                current_feature_size = out_feature
            
            self.feature_extractor_head = nn.Sequential(*fc_modules)
        else:
            self.feature_extractor_head = nn.Identity()

    def forward(self, x):
        if self.num_conv_layers > 0:
            x = x.unsqueeze(1)
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        output = self.feature_extractor_head(x)
        
        return output
    

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[], gamma=2, reduction='mean', device='cpu'):
        """
        :param alpha: A list of weight coefficients, where the weight obtained for each of the four categories is the reciprocal of the frequency of occurrence of that category
        :param gamma: Difficult samples are mined with gamma,gamma is given to the wrongly scored samples with weights turned up.
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.reciprocal(torch.tensor(alpha)).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  
        log_softmax = torch.log_softmax(pred, dim=1) 
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  
        logpt = logpt.view(-1) 
        ce_loss = -logpt 
        pt = torch.exp(logpt)  
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss 
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class MultiClassifier(nn.Module):
    def __init__(self, cell_encoder, mlp, activation=nn.ReLU):
        super(MultiClassifier, self).__init__()
        self.cell_encoder1 = cell_encoder
        self.mlp = mlp
        self.activation = activation()

    def forward(self, x):
        cell_embed1 = self.cell_encoder1(x)
        output = self.mlp(cell_embed1)
        return output

    def set_lr(self, lr: List[float]):
        optimizer_params = []
        lr_idx = 0

        optimizer_params.append({'params': self.cell_encoder1.parameters(), 'lr': lr[lr_idx]})
        lr_idx += 1

        optimizer_params.append({'params': self.mlp.parameters(), 'lr': lr[lr_idx]})
        lr_idx += 1
            
        return optimizer_params

    