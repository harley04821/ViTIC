import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData
from .base.evaluator import ProxyEvaluator
from .base.utils import *


# Embedding sources per modality
_EMB_FILES = {
    'text':  'item_cf_embeds_qwentext_8b_array.npy',   # Qwen3-Embedding-8B, dim=4096
    'vlm':   'item_cf_embeds_qwenvl_8b_array.npy',     # QwenVL-8B text+image, dim=4096
    'image': 'item_cf_embeds_qwenvl_8b_image_array.npy',  # QwenVL-8B image-only, dim=4096
}

# Variant → which modalities to load and fuse
VARIANTS = {
    'qwentext_8b':           ['text'],          # ViTIC-Text
    'qwenvl_8b':             ['vlm'],           # ViTIC-VLM
    'qwentext_qwenvlimg_8b': ['text', 'image'], # ViTIC-Dual (late fusion via concat)
}


class ModalProjector(nn.Module):
    """Projects (fused) multimodal item embeddings to the CF space.

    For single-modality input: Linear → LeakyReLU → Linear
    For dual-modality input:   same, but input_dim = sum of modality dims
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # intermediate width keeps parameter count comparable across variants
        mid_dim = max(out_dim, int(in_dim * 9 / 64) if in_dim >= 8000 else int(in_dim * 9 / 32))
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LeakyReLU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTIC_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0
        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2,
                    total=len(self.data.train_loader))
        for _, batch in pbar:
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, neg_items = batch[0], batch[1], batch[4]

            self.model.train()
            loss = self.model(users, pos_items, neg_items)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            num_batches += 1

        return [running_loss / num_batches]


class ViTIC_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)

    def add_special_model_attr(self, args):
        self.lm_model = args.lm_model
        modalities = VARIANTS[self.lm_model]
        idir = args.data_path + args.dataset + '/item_info/'

        # Load each modality and fuse via concatenation
        modal_embeds = [np.load(idir + _EMB_FILES[m]).astype(np.float32)
                        for m in modalities]
        self.item_cf_embeds = (np.concatenate(modal_embeds, axis=1)
                               if len(modal_embeds) > 1 else modal_embeds[0])

        self.modalities = modalities  # exposed for logging

        # User embedding: mean-pool over interacted items (modality-fused space)
        uid_to_items = {u: list(items) for u, items in self.train_user_list.items()}
        self.user_cf_embeds = np.array([
            self.item_cf_embeds[np.array(uid_to_items[uid])].mean(axis=0)
            for uid in sorted(uid_to_items.keys())
        ])


class ViTIC(AbstractModel):
    """Vision-Text Interaction for Cold Item Recommendation.

    Variants:
      ViTIC-Text  (--lm_model qwentext_8b)          : text-only input
      ViTIC-VLM   (--lm_model qwenvl_8b)            : vision-language input
      ViTIC-Dual  (--lm_model qwentext_qwenvlimg_8b): late fusion of text + image
    """

    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model

        item_emb = torch.tensor(data.item_cf_embeds, dtype=torch.float32).cuda(self.device)
        user_emb = torch.tensor(data.user_cf_embeds, dtype=torch.float32).cuda(self.device)
        self.register_buffer('init_item_cf_embeds', item_emb)
        self.register_buffer('init_user_cf_embeds', user_emb)

        in_dim = item_emb.shape[1]
        self.projector = ModalProjector(in_dim, self.embed_size)

    def init_embedding(self):
        pass

    def compute(self):
        # Project fused multimodal embeddings → CF space
        user_cf = self.projector(self.init_user_cf_embeds)
        item_cf = self.projector(self.init_item_cf_embeds)

        # LightGCN graph propagation
        all_emb = torch.cat([user_cf, item_cf])
        layers = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            layers.append(all_emb)

        out = torch.stack(layers, dim=1).mean(dim=1)
        return torch.split(out, [self.data.n_users, self.data.n_items])

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        u = all_users[users]
        p = all_items[pos_items]
        n = all_items[neg_items]

        if self.train_norm:
            u, p, n = F.normalize(u, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)

        pos_score = torch.sum(u * p, dim=-1)
        neg_score = torch.bmm(u.unsqueeze(1), n.permute(0, 2, 1)).squeeze(1)

        loss = -torch.log(
            torch.exp(pos_score / self.tau) /
            (torch.exp(pos_score / self.tau) + torch.exp(neg_score / self.tau).sum(dim=1))
        ).mean()
        return loss

    @torch.no_grad()
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items = self.compute()
        u = all_users[torch.tensor(users).cuda(self.device)]
        i = all_items[torch.tensor(items).cuda(self.device)]

        if self.pred_norm:
            u, i = F.normalize(u, dim=-1), F.normalize(i, dim=-1)

        return torch.matmul(u, i.T).cpu().detach().numpy()
