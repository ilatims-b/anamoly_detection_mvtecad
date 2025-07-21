# models/padim.py
import random, math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

# ---------------------------------------------------------------------- #
# Helper: feature extractor that returns intermediate layer activations  #
# ---------------------------------------------------------------------- #
class FeatureExtractor(nn.Module):
    """Freeze backbone and expose intermediate layers."""
    def __init__(self,
                 backbone_name: str = "resnet18",
                 layers: List[str] = ("layer1", "layer2", "layer3")):
        super().__init__()
        backbone = getattr(tv, backbone_name)(weights=None)           # no pre-training
        self.backbone = backbone
        self.return_layers = layers
        self._hook_handles = []
        
        # Register forward hooks for specified layers
        for name, module in backbone.named_children():
            if name in layers:
                h = module.register_forward_hook(lambda m, i, o: self._feats.append(o))
                self._hook_handles.append(h)

        # freeze params
        for p in self.backbone.parameters(): 
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feats = []
        _ = self.backbone(x)                     # hooks fill self._feats
        if len(self._feats) == 0:
            print(f"[DEBUG] No features extracted! Registered layers: {self.return_layers}")
            print(f"[DEBUG] Backbone children: {[name for name, _ in self.backbone.named_children()]}")
            raise RuntimeError("No features extracted. Check layer names and hook registration.")
        # each tensor: [B,C,H,W] – resize & concat channel-wise
        embs = [F.interpolate(f, size=self._feats[0].shape[-2:], mode="bilinear",
                              align_corners=False) for f in self._feats]
        return torch.cat(embs, dim=1)            # [B, ΣC, H, W]

    # clean up hooks when done
    def __del__(self):
        for h in self._hook_handles:
            h.remove()

# ---------------------------------------------------------------------- #
# Core PaDiM Module                                                      #
# ---------------------------------------------------------------------- #
class PaDiM(nn.Module):
    """
    Patch-Distribution Modeling (no learning, only Gaussian fitting).
    After .fit(train_loader) call, run forward(images) to obtain anomaly scores.
    """
    def __init__(self,
                 backbone: str = "resnet18",
                 layers: Tuple[str] = ("layer1", "layer2", "layer3"),
                 max_features: int = 100):
        super().__init__()
        self.extractor = FeatureExtractor(backbone, list(layers))
        
        # original channels per backbone
        orig_dims = {"resnet18": 448, "wide_resnet50_2": 1792}[backbone]
        
        # choose subset idx for random dim-reduction
        self.register_buffer("sel_idx",
                             torch.tensor(sorted(random.sample(range(orig_dims),
                                                               min(max_features, orig_dims)))))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("icov", torch.empty(0))

    # -------------------------------------------------- #
    #   Training phase: compute μ and Σ⁻¹ per patch      #
    # -------------------------------------------------- #
    @torch.no_grad()
    def fit(self, train_loader):
        self.extractor.eval()
        feats = []                          # list of [B,C,H,W]
        
        print("Extracting features from training data...")
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, dict):
                imgs = batch['image']
            else:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            imgs = imgs.to(self.sel_idx.device)
            f = self.extractor(imgs)[:, self.sel_idx]       # C'=max_features
            B, C, H, W = f.shape
            feats.append(f.permute(0,2,3,1).reshape(-1,C))  # → [B*H*W, C]
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(train_loader)}")

        feats = torch.cat(feats, 0)         # N×C
        print(f"Total features collected: {feats.shape}")

        # statistics per spatial location – reshape back
        N, C = feats.shape
        num_images = len(train_loader.dataset)
        H = W = int(math.sqrt(N // num_images))
        
        print(f"Reshaping features: N={N}, C={C}, num_images={num_images}, H={H}, W={W}")
        
        feats = feats.reshape(num_images, H*W, C)   # [num_imgs, P, C]
        
        # mean & covariance over images, per patch idx
        print("Computing statistics...")
        mean = feats.mean(0)                # [P,C]
        dif = feats - mean.unsqueeze(0)     # broadcast
        cov = torch.einsum("npc,npd->pcd", dif, dif) / (feats.shape[0] - 1)  # [P,C,C]
        
        # inverse via pseudo-inverse for stability
        print("Computing inverse covariance...")
        cov = cov.float()
        cov_cpu = cov.cpu()
        eye_cpu = torch.eye(C, device='cpu', dtype=torch.float32)
        icov_cpu = torch.linalg.pinv(cov_cpu + 1e-6 * eye_cpu).float()
        icov = icov_cpu.to(cov.device)

        self.mean = mean                    # buffers
        self.icov = icov
        
        print(f"PaDiM fitting complete. Mean shape: {self.mean.shape}, ICov shape: {self.icov.shape}")

    # -------------------------------------------------- #
    #   Inference: Mahalanobis distance per patch        #
    # -------------------------------------------------- #
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not hasattr(self, 'mean') or not hasattr(self, 'icov') or
            self.mean.numel() == 0 or self.icov.numel() == 0):
            raise RuntimeError("Model not fitted. Please call fit() first.")
            
        f = self.extractor(x)[:, self.sel_idx]             # [B,C,H,W]
        B, C, H, W = f.shape
        f = f.permute(0,2,3,1).reshape(B, H*W, C)          # [B,P,C]
        
        dist = []
        for p in range(H*W):
            diff = f[:,p,:] - self.mean[p]                 # [B,C]
            m = torch.einsum("bi,ij,bj->b", diff, self.icov[p], diff)
            dist.append(m)
        
        dist = torch.stack(dist, 1)                        # [B, P]
        score_map = dist.reshape(B, H, W)
        
        # upscale to input size if necessary:
        score_map = F.interpolate(score_map.unsqueeze(1), size=x.shape[-2:],
                                  mode="bilinear", align_corners=False).squeeze(1)
        
        return score_map       # higher = more anomalous