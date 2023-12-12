import torch
import torch.nn.functional as F
from megablocks import grouped_gemm_util as gg

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

def scaled_gelu(X, approximate="tanh"):
    # This is kinda troll lol. Scaling up to correct for variance shrinkage from GeLU.
    return 1.5876 * F.gelu(X, approximate=approximate)