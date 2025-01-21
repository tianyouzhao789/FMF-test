# @Time    : 2025/1/11 17:23
# @Author  : Tianyou Zhao
# @Email   : zhaotianyou@home.hpu.edu.cn
# @File    : module.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from fontTools.merge.util import current_time

# 辅助函数
def exists(val):
    # 判断变量是否存在（即是否为 None）
    return val is not None

def pair(t):
    # 如果输入是一个数值，将其转换为元组形式 (t, t)
    return t if isinstance(t, tuple) else (t, t)



class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class Add_cls_token(nn.Module):
    def __init__(self, dropout = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, video, current):
        bs, num_frame_patches, num_img_patch, embed_dim = video.size()
        cls_token = torch.randn(bs, num_frame_patches, 1, embed_dim, requires_grad=True)
        cls_token_current = torch.randn(bs, 1, embed_dim, requires_grad=True)
        video = torch.cat([cls_token, video], dim=2)
        current = torch.cat([cls_token_current, current], dim=1)
        return self.dropout(video), self.dropout(current)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 num_frame_patches,
                 num_image_patches,
                 embed_dim,
                 frame
                 ):
        super().__init__()
        # 可学习的位置编码
        self.video_pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, embed_dim))
        self.current_pos_embedding = nn.Parameter(torch.randn(1, frame, embed_dim))
    def forward(self, video, current):
        return video + self.video_pos_embedding, current + self.current_pos_embedding

# 定义 Attention 层
class MHSA(nn.Module):
    """
    多头自注意力机制模块。
    """
    def __init__(self, embed_dim, heads = 8, dim_head = 64, dropout = 0.):
        """
        参数:
        - dim: 输入特征维度
        - heads: 多头注意力中的头数
        - dim_head: 每个头的特征维度
        - dropout: dropout 比例
        """
        super().__init__()
        inner_dim = dim_head * heads          # 总的特征维度
        project_out = not (heads == 1 and dim_head == embed_dim)

        self.heads = heads                    # 多头数
        self.scale = dim_head ** -0.5         # 缩放因子，用于稳定注意力分数

        self.norm = nn.LayerNorm(embed_dim)         # 输入归一化
        self.attend = nn.Softmax(dim = -1)    # softmax 计算注意力权重
        self.dropout = nn.Dropout(dropout)    # dropout 防止过拟合

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias = False)  # 一次性计算 query, key, value

        # 如果需要投影到输出维度，则使用全连接层，否则直接返回
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        参数:
        - x: 输入张量，形状为 [batch_size, seq_len, dim]
        """
        x = self.norm(x)  # 归一化
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将 q, k, v 分成 3 部分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # q, k, v 的形状：[batch_size, heads, seq_len, dim_head]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算注意力分数
        # dots 形状：[batch_size, heads, seq_len, seq_len]

        attn = self.attend(dots)  # 通过 softmax 得到注意力权重
        attn = self.dropout(attn)  # 应用 dropout

        out = torch.matmul(attn, v)  # 加权求和得到输出
        out = rearrange(out, 'b h n d -> b n (h d)')  # 恢复到原始形状
        return self.to_out(out)

class MHCA(nn.Module):
    """
    多头交叉注意力机制模块。
    """
    def __init__(self, embed_dim, heads=8, dim_head=64, dropout=0.):
        """
        参数:
        - embed_dim: 输入特征维度
        - heads: 多头注意力中的头数
        - dim_head: 每个头的特征维度
        - dropout: dropout 比例
        """
        super().__init__()
        inner_dim = dim_head * heads  # 总的特征维度
        project_out = not (heads == 1 and dim_head == embed_dim)

        self.heads = heads  # 多头数
        self.scale = dim_head ** -0.5  # 缩放因子，用于稳定注意力分数

        self.norm_q = nn.LayerNorm(embed_dim)  # 对 query 归一化
        self.norm_kv = nn.LayerNorm(embed_dim)  # 对 key 和 value 归一化

        self.attend = nn.Softmax(dim=-1)  # softmax 计算注意力权重
        self.dropout = nn.Dropout(dropout)  # dropout 防止过拟合

        # Query, Key, Value 的线性映射
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(embed_dim, inner_dim * 2, bias=False)

        # 如果需要投影到输出维度，则使用全连接层，否则直接返回
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, context):
        """
        参数:
        - x: 查询张量，形状为 [batch_size, seq_len_q, dim]
        - context: 上下文张量（键值对），形状为 [batch_size, seq_len_kv, dim]
        """
        q = self.norm_q(x)  # 对查询张量归一化
        kv = self.norm_kv(context)  # 对上下文张量归一化

        # 线性映射生成 Query、Key 和 Value
        q = self.to_q(q)  # 查询张量
        k, v = self.to_kv(kv).chunk(2, dim=-1)  # 键和值

        # 将 q, k, v 重新排列为多头格式
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)  # 通过 softmax 得到注意力权重
        attn = self.dropout(attn)  # 应用 dropout

        # 使用注意力权重对 Value 加权求和
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 恢复到原始形状

        # 投影到输出维度
        return self.to_out(out)


class Patch_embedding(nn.Module):
    def __init__(self,
                 in_channels,
                 image_patch_size, # 图像patch大小
                 frame_patch_size, # 时间维度上的patch大小
                 embed_dim,       # Transformer 的输入维度
                 dilation_rate       # 膨胀采样的步长
                 ):
        super().__init__()
        dilation_rate += 1
        patch_height, patch_width = pair(image_patch_size)
        patch_dim = in_channels * patch_height * patch_width * frame_patch_size  # 每个patch的原始特征维度
        # 将视频帧划分为patch，并映射到 Transformer 输入维度
        self.video_to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)',
                p1 = patch_height,
                p2 = patch_width,
                pf = frame_patch_size
            ),                                   # 重新排列视频帧为patch, [batch_size, num_frames_patches, num_patches, patch_dim]
            nn.LayerNorm(patch_dim),            # 层归一化
            nn.Linear(patch_dim, embed_dim),          # 映射到 Transformer 的输入维度
            nn.LayerNorm(embed_dim)                   # 再次归一化
        )
        # 使用三维卷积实现膨胀采样
        self.dilated_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(frame_patch_size, patch_height, patch_width), stride=(frame_patch_size, patch_height, patch_width), dilation=(1, dilation_rate, dilation_rate)),
            Rearrange('b c f h w -> b f (h w) c')
        )
        # 将电流帧映射到 Transformer 的输入维度
        self.current_to_patch_embedding = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, Video, Current):
        #----------------输入验证----------------
        frame_size = Video.size(2)
        frame_patch_size = 2
        # 验证帧数是否可以被 frame_patch_size 整除，如果不能整除删除最后一帧
        if frame_size % frame_patch_size != 0:
            Video = Video[:, :, :-1, :, :]
            Current = Current[:, :-1, :]

        #------------- ---输入验证----------------
        video_patches_1 = self.video_to_patch_embedding(Video)
        # print(video_patches_1.shape)
        current_patches = self.current_to_patch_embedding(Current)
        # 膨胀采样
        video_dailated_patch_embedding = self.dilated_conv(Video)
        # print(video_dailated_patch_embedding.shape)
        video_patch = torch.cat([video_patches_1, video_dailated_patch_embedding], dim=2)

        # return video_patches_1, current_patches

        return video_patch , current_patches


class Tokens(nn.Module):
    def __init__(self,
                 image_pathc_size=8,
                 frame_patch_size=2,
                 embed_dim=36,
                 dilation_rate=1,
                 in_chennel=3,
                 dropout=0.
                 ):
        super().__init__()
        self.image_pathc_size = image_pathc_size
        self.frame_patch_size = frame_patch_size
        self.embed_dim = embed_dim
        self.dilation_rate = dilation_rate
        self.dropout = nn.Dropout(dropout)
        self.in_chennel = in_chennel

    def forward(self, video, current):
        patch_embedding = Patch_embedding(self.in_chennel,
                                          self.image_pathc_size,
                                          self.frame_patch_size,
                                          self.embed_dim,
                                          self.dilation_rate)
        # video_token:[batch_size, num_frames_patches, num_patches, patch_dim]
        # current_token:[batch_size, frames, patch_dim]
        _, frame, _ = current.size()
        video_patch_embedding, current_patch_embedding = patch_embedding(video, current)
        _, num_frame_patches, num_image_patches, embed_dim = video_patch_embedding.size()
        pos_embedding = PositionalEncoding(num_frame_patches, num_image_patches, embed_dim, frame)
        video_embedding_pos, current_embedding_pos = pos_embedding(video_patch_embedding, current_patch_embedding)
        add_cls_token = Add_cls_token()
        video_token, current_token = add_cls_token(video_embedding_pos, current_embedding_pos)
        return video_token, current_token


class FMFencoder(nn.Module):
    """
    FMF 编码器，用于提取视频帧和电流特征，并进行跨模态融合
    """
    def __init__(self,
                 embed_dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.
                 ):
        """
        参数:
        - dim: 输入特征维度
        - depth: Transformer 的层数
        - heads: 每层的多头数
        - dim_head: 每个头的特征维度
        - mlp_dim: 前馈网络隐藏层维度
        - dropout: dropout 比例
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)  # 层归一化
        self.layers = nn.ModuleList([])  # 保存 Transformer 层
        self.layers2 = nn.ModuleList([])  # 保存 Transformer 层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MHSA(embed_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(embed_dim, mlp_dim, dropout = dropout)
            ]))
        for _ in range(depth):
            self.layers2.append(nn.ModuleList([
                MHCA(embed_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(embed_dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x, y):
        """
        参数:
        - x: 输入张量，形状为 [batch_size, seq_len, dim]
        """
        for attn, ff in self.layers:
            # 视频帧
            x = attn(x) + x  # 残差连接 + 自注意力
            x = ff(x) + x    # 残差连接 + 前馈网络
            # 电流帧
            y = attn(y) + y  # 残差连接 + 自注意力
            y = ff(y) + y    # 残差连接 + 前馈网络
        self.norm(x)
        self.norm(y)
        for attn, ff in self.layers2:
            # 跨模态融合
            c_v = attn(x, y) + x
            c_v = ff(c_v) + c_v
            v_c = attn(y, x) + y
            v_c = ff(v_c) + v_c
        # print(c_v.shape)
        # print(v_c.shape)
        return c_v, v_c


class Classification_head(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_classes,
                 hidden_dim
                 ):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.outmudel = nn.Softmax(dim=-1)
    def forward(self, c_v, v_c):
        # print(v_c.shape)
        c_v = c_v[:, :, 0, :]
        v_c = v_c[:, 0, :]
        # print(v_c.shape)
        # print(c_v.shape)

        c_v = self.MLP(c_v)
        v_c = self.MLP(v_c)
        y = c_v + v_c
        # print(y.shape)
        return self.outmudel(y)

class FMF(nn.Module):
    def __init__(self,
                 image_pathc_size=8,
                 frame_patch_size=2,
                 embed_dim=36,
                 dilation_rate=1,
                 in_chennel=3,
                 dropout=0.,
                 depth=6,
                 heads=8,
                 dim_head=64,
                 mlp_dim=2048,
                 num_classes=2,
                 hidden_dim=1024,
                 batch_size=1
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.tokens = Tokens(image_pathc_size, frame_patch_size, embed_dim, dilation_rate, in_chennel, dropout)
        self.encoder = FMFencoder(embed_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.classification_head = Classification_head(embed_dim, num_classes, hidden_dim)

    def forward(self, video, current):
        # video_token:[batch_size, num_frames, num_patches, patch_dim]
        # current_token:[batch_size, frames, patch_dim]
        video_token, current_token = self.tokens(video, current)
        # video_token:[batch_size, num_frames*num_patches, patch_dim]
        video_token = rearrange(video_token, 'b f n d -> (b f) n d')
        # c_v:[batch_size, num_frames*num_patches, patch_dim]
        # v_c:[batch_size, frames, patch_dim]
        c_v, v_c = self.encoder(video_token, current_token)
        c_v = rearrange(c_v, '(b f) n d -> b f n d', b = self.batch_size)
        y = self.classification_head(c_v, v_c)
        return y

if __name__ == '__main__':
    video = torch.randn(1, 3, 12, 64, 64) # [batch_size, in_channels, num_frames, height, width]
    current = torch.randn(1, 12, 3) # [batch_size, frames, 3]
    model = FMF()
    out = model(video, current)
    print(out)

