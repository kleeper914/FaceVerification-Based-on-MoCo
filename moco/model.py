import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F

class MoCo(nn.Module):
    '''
    MoCo: https://arxiv.org/abs/1911.05722
    '''
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, dropout=0.2):
        '''
        @param:
            base_encoder: base encoder network
            dim: 特征维度, feature_dimension
            K: queue size, 负样本的数量
            m: momentum for updating key encoder
            T: temperature parameter
            mlp: 是否使用mlp
        '''
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # encoders, num_classes是fc的输出维度
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                self.encoder_k.fc
            )
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''
        更新key encoder
        '''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        '''
        出队入队
        '''
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        '''
        打乱batch
        '''
        # random shuffle
        idx_shuffle = torch.randperm(x.shape[0])
        # idx_unshuffle = torch.argsort(idx_shuffle)
        idx_unshuffle = torch.empty_like(idx_shuffle, device=x.device)
        idx_unshuffle[idx_shuffle] = torch.arange(x.shape[0], device=x.device)
        x = x[idx_shuffle]
        return x, idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        '''
        恢复batch
        '''
        return x[idx_unshuffle]
    
    def forward(self, im_q, im_k):
        '''
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        '''
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1) # q: NxC
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # l_pos = F.cosine_similarity(q, k, dim=1).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # l_neg = F.cosine_similarity(q.unsqueeze(1), self.queue.clone().detach().T.unsqueeze(0), dim=2)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return logits, labels

@torch.no_grad()
def concat_all_gather(tensor):
    '''
    所有进程的tensor拼接
    '''
    return tensor