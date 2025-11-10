import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, drop=0.0, bias=True):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features, bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, d_emb, num_heads, qkv_bias=False, proj_bias=False):

        super().__init__()

        self.num_heads = num_heads
        self.d_head = d_emb // num_heads
        self.qkv = nn.Linear(d_emb, 3 * num_heads * self.d_head, bias=qkv_bias)
        self.proj = nn.Linear(d_emb, d_emb, bias=proj_bias)

    def forward(self, x):
        B, T, _ = x.shape

        # (B, T, 3, H, d)
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # (B, T, H, d)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = (q @ k.transpose(-1, -2)) * self.d_head**-0.5
        x = nn.functional.softmax(x, dim=-2)
        x = x @ v

        x = x.transpose(1, 2).reshape(B, T, -1)

        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, d_emb, num_heads, qkv_bias=False, proj_bias=False):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_emb)
        self.attn = SelfAttention(
            d_emb, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias
        )

        self.norm2 = nn.LayerNorm(d_emb)
        self.mlp = MLP(d_emb, d_emb, hidden_features=2 * d_emb)

    def forward(self, x):
        s1 = x
        x = self.norm1(x)
        x = self.attn(x)
        x += s1
        s2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += s2

        return x


class VIT(nn.Module):
    def __init__(
        self,
        patch_size=8,
        d_emb=128, 
        num_heads=8, 
        seq_len=256, 
        layers=8
    ):
        super().__init__()

        self.patch_emb = nn.Conv2d(3, d_emb, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Embedding(seq_len + 1, d_emb)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_emb))
        
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_emb=d_emb, num_heads=num_heads) for _ in range(layers)]
        )

        self.final_norm = nn.LayerNorm(d_emb)

    def forward(self, x):
        e = self.patch_emb(x).flatten(start_dim=-2).transpose(1,2)
        e = torch.cat(
            [self.class_token.expand(e.shape[0], -1, -1), e],
            dim=1
        )
        p = self.pos_emb(torch.arange(e.shape[-2]).to(x.device))[None]
        z = self.blocks(e + p)
        z = self.final_norm(z)

        return z[:, 0]


def main():
    pass


if __name__ == "__main__":

    main()
