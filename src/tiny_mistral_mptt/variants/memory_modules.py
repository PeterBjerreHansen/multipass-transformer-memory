"""Shared memory emission and single-record merger modules."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from tiny_mistral.modeling import MistralRMSNorm


class MemoryWriter(nn.Module):
    """Learn a D-to-D memory record from a final normalized hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("writer input must be [B,T,D]")
        return self.proj(hidden_states)


class AdaptiveRecirculationController(nn.Module):
    """Predict token- and feature-wise recirculation coefficients.

    This follows Mozer et al.'s conditional vector alpha/beta controller: a
    two-hidden-layer GELU MLP consumes the concatenated source and destination
    residual streams and predicts independent sigmoid-bounded coefficient
    vectors. The zero output-weight initialization makes the controller start
    at the fixed convex mixture and lets the coefficient head learn first.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        initial_alpha: float,
        initialization_seed: int,
    ):
        super().__init__()
        input_size = 2 * int(hidden_size)
        hidden_size = int(hidden_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.input_norm = nn.LayerNorm(input_size)
            self.hidden_1 = nn.Linear(input_size, hidden_size)
            self.hidden_2 = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, 2 * hidden_size)
            nn.init.xavier_uniform_(self.hidden_1.weight)
            nn.init.zeros_(self.hidden_1.bias)
            nn.init.xavier_uniform_(self.hidden_2.weight)
            nn.init.zeros_(self.hidden_2.bias)
            # At initialization, the adaptive controller is the existing
            # fixed-alpha recirculation rule. Sigmoid cannot represent exact
            # endpoints, so clamp only for the initialization logit.
            safe_alpha = min(max(float(initial_alpha), 1e-6), 1.0 - 1e-6)
            safe_beta = min(max(1.0 - float(initial_alpha), 1e-6), 1.0 - 1e-6)
            alpha_logit = math.log(safe_alpha / (1.0 - safe_alpha))
            beta_logit = math.log(safe_beta / (1.0 - safe_beta))
            nn.init.zeros_(self.output.weight)
            nn.init.constant_(self.output.bias[:hidden_size], alpha_logit)
            nn.init.constant_(self.output.bias[hidden_size:], beta_logit)

    def forward(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((source, destination), dim=-1)
        hidden = F.gelu(self.hidden_1(self.input_norm(features)))
        hidden = F.gelu(self.hidden_2(hidden))
        logits = self.output(hidden)
        alpha_logits, beta_logits = logits.chunk(2, dim=-1)
        return torch.sigmoid(alpha_logits), torch.sigmoid(beta_logits)


def norm_match(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    source_norm = torch.linalg.vector_norm(source.float(), ord=2, dim=-1, keepdim=True)
    destination_norm = torch.linalg.vector_norm(destination.float(), ord=2, dim=-1, keepdim=True)
    return source * (destination_norm / source_norm.clamp_min(1e-12)).to(source.dtype)


class RecirculationMerger(nn.Module):
    """Adaptive norm-matched alpha/beta mixing of one emitted memory record."""

    def __init__(self, hidden_size: int, *, initialization_seed: int):
        super().__init__()
        self.controller = AdaptiveRecirculationController(
            hidden_size, initial_alpha=0.1, initialization_seed=initialization_seed
        )

    def forward(self, destination: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        matched = norm_match(memory, destination)
        alpha, beta = self.controller(memory, destination)
        return alpha * matched + beta * destination


class ProjectedResidualMerger(nn.Module):
    """Identity-initialized, conditionally gated projection of one memory record.

    The destination retains an exact identity path. The projection learns first;
    its nonzero updates then let gradients reach the gate and memory writer.
    There is no second zero-initialized scalar gate to block that first update.
    """

    def __init__(self, hidden_size: int, *, eps: float, initialization_seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.destination_norm = MistralRMSNorm(hidden_size, eps=eps)
            self.memory_norm = MistralRMSNorm(hidden_size, eps=eps)
            self.gate = nn.Linear(2 * hidden_size, hidden_size)
            self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            nn.init.zeros_(self.projection.weight)

    def forward(self, destination: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        gate_input = torch.cat(
            (self.destination_norm(destination), normalized_memory), dim=-1
        )
        return destination + torch.sigmoid(self.gate(gate_input)) * self.projection(
            normalized_memory
        )
