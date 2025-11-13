import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphConv(nn.Module):
    """
    简单的超图卷积层：Node -> Hyperedge -> Node

    输入:
        X: 节点特征, shape = [N, d_in]
        H: incidence 矩阵, shape = [N, M] (0/1 或 实数)

    输出:
        X_out: 更新后的节点特征, shape = [N, d_out]
    """
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True, activation=F.relu):
        super().__init__()

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        X: [N, d_in]
        H: [N, M]
        """
        # 节点度: 每个节点属于多少个超边
        # dv: [N, 1]
        dv = H.sum(dim=1, keepdim=True).clamp(min=1.0)
        de = H.sum(dim=0, keepdim=True).clamp(min=1.0)

        # ------- Node -> Hyperedge 聚合 -------
        X_norm = X / dv              # [N, d_in]
        Xe = H.t() @ X_norm
        Xe = Xe / de.t()             # [M, d_in]
        # ------- Hyperedge -> Node 聚合 -------
        Xv = H @ Xe                

        return Xv