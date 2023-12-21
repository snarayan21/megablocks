import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from megablocks import grouped_gemm_util as gg
import numpy as np
from typing import Any, Optional

# Scaling code based on Unit Scaling paper:
# https://arxiv.org/abs/2303.11257
class ScaledGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, alpha, beta):
        ctx.save_for_backward(torch.tensor(beta, dtype=X.dtype))
        return alpha * X

    @staticmethod
    def backward(ctx, grad_Y):
        beta, = ctx.saved_tensors
        return beta * grad_Y, None, None
    
def scaled(X, alpha=1, beta=1):
    # Forward: Y = X * alpha
    # Backward: grad_X = grad_Y * beta
    return ScaledGrad.apply(X, alpha, beta)

def scaled_gmm(a, b, batch_sizes, trans_b=False, constrain_a=True, constrain_b=True):
    (m, k) = a.shape
    if trans_b:
        n = b.shape[-2]
    else:
        n = b.shape[-1]
    alpha = k ** -(1/2)
    beta_a = n ** -(1/2)
    beta_b = m ** -(1/2)
    if constrain_a and constrain_b:
        beta_a = alpha
        beta_b = alpha
    elif constrain_a:
        beta_a = alpha
    elif constrain_b:
        beta_b = alpha

    # Scale down the matmul inputs, but only on the backwards pass
    a = scaled(a, beta=beta_a)
    b = scaled(b, beta=beta_b)
    
    # Scale down the matmul result, but only on the forwards pass
    return scaled(gg.ops.gmm(a, b, batch_sizes, trans_b=trans_b), alpha=alpha)

def scaled_gmm_custom_bwd(a, b, batch_sizes, trans_b=False, a_bwd_scale=None, b_bwd_scale=None):
    (m, k) = a.shape
    if trans_b:
        n = b.shape[-2]
    else:
        n = b.shape[-1]
    alpha = k ** -(1/2)
    beta_a = alpha if a_bwd_scale is None else a_bwd_scale
    beta_b = alpha if b_bwd_scale is None else b_bwd_scale

    # Scale down the matmul inputs, but only on the backwards pass
    a = scaled(a, beta=beta_a)
    b = scaled(b, beta=beta_b)
    
    # Scale down the matmul result, but only on the forwards pass
    return scaled(gg.ops.gmm(a, b, batch_sizes, trans_b=trans_b), alpha=alpha)

def scaled_matmul(a, b, constrain_a=True, constrain_b=True):
    (m, k), (_, n) = a.shape, b.shape
    alpha = k ** -(1/2)
    beta_a = n ** -(1/2)
    beta_b = m ** -(1/2)
    if constrain_a and constrain_b:
        beta_a = alpha
        beta_b = alpha
    elif constrain_a:
        beta_a = alpha
    elif constrain_b:
        beta_b = alpha

    # Scale down the matmul inputs, but only on the backwards pass
    a = scaled(a, beta=beta_a)
    b = scaled(b, beta=beta_b)
    
    # Scale down the matmul result, but only on the forwards pass
    return scaled(torch.matmul(a, b), alpha=alpha)

def scaled_matmul_custom_bwd(a, b, a_bwd_scale=None, b_bwd_scale=None):
    (m, k), (_, n) = a.shape, b.shape
    alpha = k ** -(1/2)
    beta_a = alpha if a_bwd_scale is None else a_bwd_scale
    beta_b = alpha if b_bwd_scale is None else b_bwd_scale

    # Scale down the matmul inputs, but only on the backwards pass
    a = scaled(a, beta=beta_a)
    b = scaled(b, beta=beta_b)
    
    # Scale down the matmul result, but only on the forwards pass
    return scaled(torch.matmul(a, b), alpha=alpha)

def scaled_linear_custom_bwd(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    a_bwd_scale: Optional[float] = None,
    b_bwd_scale: Optional[float] = None
) -> Tensor:
    k = input.shape[-1]
    m = input.shape[-2]
    # nn.Linear weights are stored as fan_out x fan_in.
    n = weight.shape[-2]
    if k != weight.shape[-1]:
        raise ValueError(f"Scaled linear shared dimensions must match. \
                         Got {k} and {weight.shape[-2]} instead.")

    alpha = k**-(1/2)
    beta_a = n**-(1/2)
    beta_b_weight = beta_b_bias = m**-(1/2)

    beta_a = alpha if a_bwd_scale is None else a_bwd_scale
    beta_b_weight = alpha if b_bwd_scale is None else b_bwd_scale
    beta_b_bias = alpha if b_bwd_scale is None else b_bwd_scale

    input = scaled(input, beta=beta_a)
    weight = scaled(weight, beta=beta_b_weight)
    bias = scaled(bias, beta=beta_b_bias) if bias is not None else None
    return scaled(F.linear(input, weight, bias), alpha=alpha)

class ScaledLinearCustomBwd(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
        a_bwd_scale: Optional[float] = None,
        b_bwd_scale: Optional[float] = None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.a_bwd_scale = a_bwd_scale
        self.b_bwd_scale = b_bwd_scale

    def forward(self, input: Tensor) -> Tensor:
        return scaled_linear_custom_bwd(input, self.weight, self.bias, self.a_bwd_scale, self.b_bwd_scale)

def scaled_gelu(X, approximate="tanh"):
    # This is kinda troll lol. Scaling up to correct for variance shrinkage from GeLU.
    return (1/0.588) * F.gelu(X, approximate=approximate)

def top_k_softmax_std(dim: int = 2048, top_k = 4):
    return (1/dim)*(np.log(dim/top_k)+1)