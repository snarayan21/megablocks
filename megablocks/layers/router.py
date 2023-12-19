from megablocks.layers import common
from megablocks.layers.arguments import Arguments
from megablocks.layers.scaling import scaled_matmul_custom_bwd, scaled, top_k_softmax_std
import torch
import torch.nn.functional as F


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        if args.unit_scaling:
            self.layer = torch.nn.Parameter(torch.empty(
                args.hidden_size,
                args.moe_num_experts,
                dtype=common.dtype(args),
                device=args.device))
            args.init_method(self.layer)
        else:
            self.layer = torch.nn.Linear(
                args.hidden_size,
                args.moe_num_experts,
                bias=False,
                dtype=common.dtype(args),
                device=args.device)
            args.init_method(self.layer.weight)

        num_experts = self.args.moe_num_experts
        top_k_scaling_param = top_k_softmax_std(dim=args.moe_num_experts, top_k=args.moe_top_k)
        logits_grad_scale = (num_experts**4/(2*num_experts**2 - 4*num_experts + 8))**0.5
        experts_weights_grad_scale = (1/args.hidden_size)**0.5*(1/args.residual_coeff)
        weighted_experts_scale = 1/(top_k_scaling_param*(2*args.moe_top_k)**0.5)*(1/0.867)
        self.router_grad_scale = ((1/args.ddp_tokens)**0.5)*(1/0.922)*logits_grad_scale*experts_weights_grad_scale/weighted_experts_scale

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)


    def forward(self, x):
        
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        if self.args.unit_scaling:
            scores = scaled_matmul_custom_bwd(x.view(-1, x.shape[-1]), self.layer, b_bwd_scale=self.router_grad_scale)
            scores = F.softmax(scores, dim=-1)
        else:
            scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
        expert_weights, expert_indices = self._top_k(scores)

        # Normalize expert weights by euclidean norm to preserve output variance.
        # expert_weights = torch.nn.functional.normalize(expert_weights, p=2, dim=-1)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, expert_weights, expert_indices
