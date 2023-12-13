from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).
                beta_1, beta_2 = group["betas"]
                
                if "m" not in state:
                    state["m"], state["v"], state["t"] = 0, 0, 0
                
                m, v, t = state["m"], state["v"], state["t"]
                t += 1
                
                # get gradients w.r.t. stochastic objective function at timestep t
                # g_t <- gradient of f_t( θ(t - 1) )
                
                # update biased first moment estimate
                # m_t <- B_1 * m_(t - 1) + (1 - B_1) * g_t
                m = beta_1 * m + (1 - beta_1) * grad
                
                # update biased second raw moment estimate
                # v_t <- B_2 * v_(t - 1) + (1 - B_2) * g^2
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                
                if group["correct_bias"]:
                    # # compute bias-corrected first moment estimate
                    # # m^_t <- m_t / (1 - B_1^t)
                    # m_hat = m / (1 - beta_1 ** t)
                    
                    # # compute bias-corrected second raw moment estimate
                    # # v^_t <- v_t / (1 - B_2^t)
                    # v_hat = v / (1 - beta_2 ** t)
                    
                    # θ_t <- θ_(t - 1) - α * m_hat / (sqrt(v_hat) + ϵ)
                    # p.data = p.data - (alpha * m_hat) / (torch.sqrt(v_hat) + group["eps"])
                    
                    # efficient version
                    alpha_t = alpha * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                    p.data = p.data - alpha_t * m / (torch.sqrt(v) + group["eps"])
                
                else:
                    # θ_t <- θ_(t - 1) - α * m / (sqrt(m) + ϵ)
                    p.data = p.data - (alpha * m) / (torch.sqrt(v) + group["eps"])
                
                p.data = p.data - alpha * group["weight_decay"] * p.data
                
                state["m"], state["v"], state["t"] = m, v, t
 
        return loss
