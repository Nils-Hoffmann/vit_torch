from dataclasses import dataclass
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    num_layers: int = 12
    image_res: tuple = (224, 224, 3)
    patch_dim: int = 16 * 16 * 3
    patch_size: int = 16
    ffn_dim: int = 768 * 4
    embedding_dim: int = 768
    num_heads: int = 12
    dropout_rate: float = 0.1
    num_classes: int = 10



class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ffn_dim),
            nn.ReLU(inplace=True), # TODO: measure performance impact
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.ffn_dim, config.embedding_dim)
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.embedding_dim, eps=1e-5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x) # todo: autocasting rules?
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm))
        return x
    
class ViT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        H, W, C, = config.image_res
        num_tokens = (H // config.patch_size) * (W // config.patch_size) + 1
        self.patching = image_patching(config.patch_size)

        self.linear_proj = nn.Linear(config.patch_dim, config.embedding_dim)

        self.class_embedding = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, config.embedding_dim))

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])

        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

        self.apply(_init_weights)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patching(img)
        x = self.linear_proj(x)

        cls = self.class_embedding.expand(x.shape[0], -1,  -1) # Batch, 1, Embedding
        x = torch.cat([cls, x], dim=1)  # batch (1 + tokens) embedding

        x = x + self.pos_embedding

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)

        logits = self.classifier(x[:, 0, :]) # logits only for the cls token
        return logits

def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
        


    
def image_patching(patch_size: int) -> typing.Callable[[torch.Tensor], torch.Tensor]:
    def _patching(img: torch.Tensor) -> torch.Tensor:
        # Got HHWC input, move to torchs channel first
        if img.ndim == 4 and img.shape[1] != 3:
            img = img.permute(0, 3, 1, 2).contiguous()
        patches = F.unfold(img, patch_size, stride=patch_size) # Batch, p * p * c, length
        return patches.transpose(1, 2).contiguous() # batch, length, patch_dim
    return _patching