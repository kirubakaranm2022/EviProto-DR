import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

K = 5


class EviProtoDR(nn.Module):
    def __init__(self, pretrained=True, beta=0.99, embed_dim=256):
        super().__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        n = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier = nn.Identity()
        self.project = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, embed_dim))
        self.ce_head = nn.Linear(n, K)  # auxiliary CE head used for L_CE
        self.evidence = nn.Sequential(nn.Linear(K, 64), nn.ReLU(), nn.Linear(64, K))
        self.register_buffer("prototypes", torch.zeros(K, embed_dim))
        self.register_buffer("initialized", torch.zeros(K, dtype=torch.bool))
        self.beta = beta

    def forward(self, images):
        h = self.backbone(images)
        # Heads, distances and Dirichlet quantities always run in float32,
        # even when the backbone runs under bf16 autocast.
        with torch.autocast(device_type=images.device.type, enabled=False):
            h = h.float()
            z = F.normalize(self.project(h), dim=1)
            d = torch.cdist(z, self.prototypes.float())
            alpha = F.softplus(self.evidence(d)) + 1.0
            s = alpha.sum(1, keepdim=True)
            prob = alpha / s
            return dict(z=z, d=d, alpha=alpha, prob=prob, pred=prob.argmax(1),
                        u=K / s.squeeze(1), logits=self.ce_head(h))

    @torch.no_grad()
    def update_prototypes(self, z, y):
        """Eq. (3) EMA update, vectorised over grades (no per-class host syncs)."""
        z = z.detach().float()
        onehot = F.one_hot(y, K).to(z.dtype)
        counts = onehot.sum(0)
        present = counts > 0
        means = F.normalize(onehot.T @ z / counts.clamp_min(1)[:, None], dim=1)
        ema = F.normalize(self.beta * self.prototypes + (1 - self.beta) * means, dim=1)
        new = torch.where(self.initialized[:, None], ema, means)
        self.prototypes.copy_(torch.where(present[:, None], new, self.prototypes))
        self.initialized |= present

    @torch.no_grad()
    def initialize_from_means(self, sums, counts):
        present = counts > 0
        if not bool(present.all()):
            raise ValueError("Missing grade in first training epoch; cannot initialise all prototypes")
        self.prototypes.copy_(F.normalize(sums / counts[:, None], dim=1))
        self.initialized.fill_(True)
