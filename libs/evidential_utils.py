"""
Evidential deep learning loss functions.

Regression:  Normal-Inverse-Gamma (NIG) loss — Amini et al. (2020)
Classification: Dirichlet-based loss — Sensoy et al. (2018)

Both implementations follow the formulations used in:
  Schreck et al. (2024), "Evidential Deep Learning for Predicting Uncertainty
  in Earth System Science Applications", AI for Earth Systems.
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Regression — Normal-Inverse-Gamma (NIG)
# ---------------------------------------------------------------------------

def nig_nll(y, gamma, nu, alpha, beta):
    """Negative log-likelihood of the NIG predictive distribution.

    The marginal distribution of y under a NIG prior is a Student-t.
    This is its negative log-likelihood (Amini et al. 2020, Eq. 5).

    All inputs are (batch,) tensors; nu, alpha, beta must be positive.
    """
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * (math.log(math.pi) - torch.log(nu.clamp(min=1e-8)))
        - alpha * torch.log(omega.clamp(min=1e-8))
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll.mean()


def nig_reg(y, gamma, nu, alpha):
    """Evidence regularizer (Amini et al. 2020, Eq. 9).

    Penalises placing high evidence (small uncertainty) on points where the
    predicted mean gamma is far from the true value y.
    """
    error = torch.abs(y - gamma)
    evidence = 2.0 * nu + alpha
    return (error * evidence).mean()


def evidential_regression_loss(pred, y, coeff=0.01):
    """Full NIG evidential regression loss: NIG-NLL + coeff * regularization.

    Args:
        pred:  raw model output of shape (batch, 4) — columns are the
               unconstrained (gamma_raw, nu_raw, alpha_raw, beta_raw).
        y:     ground-truth targets, shape (batch,).
        coeff: weight for the evidence regularization term (lambda in paper).

    Returns:
        loss:  scalar loss for backprop.
        gamma, nu, alpha, beta: constrained NIG parameters, each (batch,).
    """
    gamma = pred[:, 0]                             # mean — unconstrained
    nu    = F.softplus(pred[:, 1]) + 1e-4         # keep away from 0: log(nu) in NIG NLL explodes otherwise
    alpha = F.softplus(pred[:, 2]) + 1.0          # > 1  (needed for finite variance)
    beta  = F.softplus(pred[:, 3]) + 1e-4         # keep away from 0: same reason

    loss = nig_nll(y, gamma, nu, alpha, beta) + coeff * nig_reg(y, gamma, nu, alpha)
    return loss, gamma, nu, alpha, beta


def nig_uncertainty(nu, alpha, beta):
    """Decompose NIG parameters into aleatoric and epistemic uncertainty.

    Aleatoric (data noise):   E[sigma^2]     = beta / (alpha - 1)
    Epistemic (model uncert): Var[mu]/E[mu]  = beta / (nu * (alpha - 1))

    Both are (batch,) tensors.
    """
    denom     = (alpha - 1.0).clamp(min=1e-8)
    aleatoric = beta / denom
    epistemic = beta / (nu.clamp(min=1e-8) * denom)
    return aleatoric, epistemic


# ---------------------------------------------------------------------------
# Classification — Dirichlet
# ---------------------------------------------------------------------------

def kl_dirichlet(alpha, num_classes):
    """Analytical KL(Dir(alpha) || Dir(ones)) averaged over the batch.

    alpha: (batch, num_classes), all entries >= 1.
    """
    S  = alpha.sum(dim=1, keepdim=True)
    K  = torch.tensor(float(num_classes), device=alpha.device)
    kl = (
        torch.lgamma(S)
        - torch.lgamma(K)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(
            dim=1, keepdim=True
        )
    )
    return kl.mean()


def evidential_classification_loss(pred, y, num_classes, coeff=0.01,
                                   epoch=0, warmup_epochs=10):
    """Evidential classification loss (Sensoy et al. 2018).

    Uses Type-II MLE (Bayes risk for cross-entropy) plus a KL regularization
    term that is linearly annealed from 0 to coeff over warmup_epochs.

    Args:
        pred:          raw model logits, shape (batch, num_classes).
        y:             integer class labels, shape (batch,).
        num_classes:   number of output classes.
        coeff:         maximum weight for the KL regularization term.
        epoch:         current training epoch (0-indexed), used for annealing.
        warmup_epochs: number of epochs over which to anneal the KL weight.

    Returns:
        loss:      scalar loss for backprop.
        alpha:     Dirichlet parameters (batch, num_classes), all >= 1.
        S:         Dirichlet strength sum(alpha), shape (batch, 1).
        log_probs: log(alpha/S) — passing this to evaluate_classification_multi
                   is correct because softmax(log(p)) == p when p sums to 1.
    """
    evidence = F.softplus(pred)               # (batch, num_classes), > 0
    alpha    = evidence + 1.0                 # (batch, num_classes), >= 1
    S        = alpha.sum(dim=1, keepdim=True) # (batch, 1)

    # One-hot encode targets
    y_onehot = F.one_hot(y, num_classes=num_classes).float()

    # Type-II MLE: expected cross-entropy under the Dirichlet
    loss_fit = (
        y_onehot * (torch.digamma(S) - torch.digamma(alpha))
    ).sum(dim=1).mean()

    # KL regularization: alpha_tilde removes evidence assigned to the correct
    # class so the penalty only discourages evidence on wrong classes.
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = kl_dirichlet(alpha_tilde, num_classes)

    # Linear annealing: KL weight grows from 0 to coeff over warmup_epochs
    anneal = min(1.0, (epoch + 1) / max(warmup_epochs, 1))
    loss = loss_fit + coeff * anneal * kl

    # log(alpha/S): softmax of this equals alpha/S exactly, making it
    # compatible with evaluate_classification_multi which applies softmax.
    log_probs = torch.log((alpha / S).clamp(min=1e-8))

    return loss, alpha, S, log_probs
