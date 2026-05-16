"""
Delta SAE Training Script
LoRA × Mechanistic Interpretability Research
=============================================
Trains one SAE per (rank, layer) combination on h_delta activations.
Total: 4 ranks × 6 layers = 24 SAEs.

Each SAE learns a sparse feature dictionary specifically for the
adapter's contribution to the residual stream at that layer.

Architecture:
    Standard ReLU SAE with L1 sparsity penalty
    Encoder: Linear(d_model, d_sae) + ReLU
    Decoder: Linear(d_sae, d_model, bias=False), columns normalised to unit norm
    Loss: MSE reconstruction + L1 sparsity

"""

import os
import gc
import json
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# ── Dirs ──────────────────────────────────────────────────────────────────────
ACTIVATIONS_ROOT = "./activations_v2"
SAE_ROOT         = "./delta_saes"
os.makedirs(SAE_ROOT, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SAE_ROOT, "train_sae.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANKS         = [4, 8, 16, 32]   # LoRA ranks to train on
TARGET_LAYERS = [5, 10, 18, 22, 32, 38]
D_MODEL       = 3584        # Gemma-2-9b hidden size
D_SAE         = 16384       # 4.6x expansion — matches Gemma Scope width

# Training hyperparameters
LEARNING_RATE = 1e-3
L1_COEFF      = 0.15       # sparsity penalty — tune if features collapse or too dense
BATCH_SIZE    = 512         # number of token vectors per batch
N_EPOCHS      = 10           # passes over the data
LOG_EVERY     = 200         # log every N steps
SEED          = 42

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
log.info(f"Device: {DEVICE}")


# ── SAE Architecture ──────────────────────────────────────────────────────────
class SparseAutoencoder(nn.Module):
    """
    Standard ReLU SAE.

    Forward pass:
        x_centred  = x - b_dec          (pre-centre input)
        z          = ReLU(W_enc @ x_centred + b_enc)   (sparse features)
        x_hat      = W_dec @ z + b_dec  (reconstruction)

    Loss:
        mse  = ||x - x_hat||^2  (reconstruction)
        l1   = L1_COEFF * ||z||_1  (sparsity)
        loss = mse + l1
    """
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae   = d_sae

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder (no bias on weights — bias handled separately)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialise with Kaiming uniform
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        # Normalise decoder columns to unit norm at init
        self._normalise_decoder()

    def _normalise_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data = self.W_dec.data / norms

    def encode(self, x):
        # x: [batch, d_model]
        x_centred = x - self.b_dec
        z = F.relu(x_centred @ self.W_enc + self.b_enc)
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z    = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def loss(self, x, z, x_hat, l1_coeff):
        mse  = F.mse_loss(x_hat, x, reduction="mean")
        l1   = l1_coeff * z.abs().mean()
        return mse + l1, mse.item(), l1.item()


# ── Load h_delta for one (rank, layer) ────────────────────────────────────────
def load_delta_vectors(rank, layer_idx):
    path = os.path.join(ACTIVATIONS_ROOT, f"r{rank}", f"layer_{layer_idx}_h_delta.pt")
    delta = torch.load(path, map_location="cpu")
    flat  = delta.float() if delta.dim() == 2 else delta.reshape(-1, delta.shape[-1]).float()

    # Filter out padding tokens (near-zero delta norm)
    norms = flat.norm(dim=-1)
    threshold = norms.mean() * 0.05   # keep vectors with norm > 5% of mean
    mask = norms > threshold
    flat = flat[mask]
    log.info(f"  Filtered: {mask.sum().item():,} / {len(mask):,} vectors kept (removed padding)")

    # Normalise by RMS
    rms  = flat.norm(dim=-1).mean()
    flat = flat / rms

    log.info(f"  Post-norm mean norm: {flat.norm(dim=-1).mean():.4f}")
    return flat, rms.item()

# ── Train one SAE ─────────────────────────────────────────────────────────────
def train_sae(rank, layer_idx, delta_vectors):
    """
    Trains a single SAE on delta_vectors [N, d_model].
    Returns the trained SAE and training log.
    """
    torch.manual_seed(SEED)
    N = delta_vectors.shape[0]
    log.info(f"  Training SAE on {N:,} vectors | d_model={D_MODEL} | d_sae={D_SAE}")
    log.info(f"  Epochs={N_EPOCHS} | batch={BATCH_SIZE} | lr={LEARNING_RATE} | l1={L1_COEFF}")

    dataset    = TensorDataset(delta_vectors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    n_batches  = len(dataloader)
    log.info(f"  Batches per epoch: {n_batches}")

    sae       = SparseAutoencoder(D_MODEL, D_SAE).to(DEVICE)
    optimiser = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)

    train_log = {
        "loss"     : [],
        "mse"      : [],
        "l1"       : [],
        "l0"       : [],   # mean number of active features per token
        "epoch_end": [],
    }

    global_step = 0

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        epoch_mse  = 0.0
        epoch_l1   = 0.0
        epoch_l0   = 0.0

        for (batch,) in dataloader:
            batch = batch.to(DEVICE)

            optimiser.zero_grad()
            z, x_hat  = sae(batch)
            loss, mse, l1 = sae.loss(batch, z, x_hat, L1_COEFF)
            loss.backward()
            optimiser.step()

            # Re-normalise decoder columns after each step
            sae._normalise_decoder()

            # L0 = mean number of active features (z > 0)
            l0 = (z > 0).float().sum(dim=-1).mean().item()

            epoch_loss += loss.item()
            epoch_mse  += mse
            epoch_l1   += l1
            epoch_l0   += l0

            if global_step % LOG_EVERY == 0:
                log.info(
                    log.info(
                         f"  [r={rank} | layer={layer_idx} | epoch={epoch+1} | step={global_step}] "
                         f"loss={loss.item():.6f} | mse={mse:.6f} | l1={l1:.6f} | l0={l0:.1f}"
)
                )
                train_log["loss"].append(round(loss.item(), 6))
                train_log["mse"].append(round(mse, 6))
                train_log["l1"].append(round(l1, 6))
                train_log["l0"].append(round(l0, 2))

            global_step += 1

        # Epoch summary
        avg_loss = epoch_loss / n_batches
        avg_mse  = epoch_mse  / n_batches
        avg_l1   = epoch_l1   / n_batches
        avg_l0   = epoch_l0   / n_batches

        log.info(
            f"  ── Epoch {epoch+1}/{N_EPOCHS} complete | "
            f"avg_loss={avg_loss:.4f} | avg_mse={avg_mse:.4f} | "
            f"avg_l1={avg_l1:.4f} | avg_l0={avg_l0:.1f}"
        )
        train_log["epoch_end"].append({
            "epoch"   : epoch + 1,
            "avg_loss": round(avg_loss, 6),
            "avg_mse" : round(avg_mse, 6),
            "avg_l1"  : round(avg_l1, 6),
            "avg_l0"  : round(avg_l0, 2),
        })

    log.info(f"  Training complete. Final avg_l0={avg_l0:.1f} active features/token.")
    return sae, train_log


# ── Save SAE ──────────────────────────────────────────────────────────────────
def save_sae(rank, layer_idx, sae, train_log, rms_scale):
    out_dir = os.path.join(SAE_ROOT, f"r{rank}", f"layer_{layer_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # Weights
    weights_path = os.path.join(out_dir, "sae_weights.pt")
    torch.save({
        "W_enc": sae.W_enc.detach().cpu(),
        "b_enc": sae.b_enc.detach().cpu(),
        "W_dec": sae.W_dec.detach().cpu(),
        "b_dec": sae.b_dec.detach().cpu(),
    }, weights_path)

    # Config
    config = {
        "rank"         : rank,
        "layer"        : layer_idx,
        "d_model"      : D_MODEL,
        "d_sae"        : D_SAE,
        "learning_rate": LEARNING_RATE,
        "l1_coeff"     : L1_COEFF,
        "batch_size"   : BATCH_SIZE,
        "n_epochs"     : N_EPOCHS,
        "seed"         : SEED,
        "device"       : DEVICE,
        "rms_scale": rms_scale,
        "activations"  : os.path.join(ACTIVATIONS_ROOT, f"r{rank}", f"layer_{layer_idx}_h_delta.pt"),
        "final_avg_l0" : train_log["epoch_end"][-1]["avg_l0"],
        "final_avg_mse": train_log["epoch_end"][-1]["avg_mse"],
        "saved_at"     : datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "sae_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training log
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    size_mb = os.path.getsize(weights_path) / 1e6
    log.info(f"  ✅ SAE saved to {out_dir}/ ({size_mb:.1f}MB)")
    return config


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║       Delta SAE Training — LoRA × Mech Interp           ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Ranks         : {RANKS}")
    log.info(f"Layers        : {TARGET_LAYERS}")
    log.info(f"D_MODEL       : {D_MODEL}")
    log.info(f"D_SAE         : {D_SAE}")
    log.info(f"L1_COEFF      : {L1_COEFF}")
    log.info(f"N_EPOCHS      : {N_EPOCHS}")
    log.info(f"Device        : {DEVICE}")
    log.info(f"Total SAEs    : {len(RANKS) * len(TARGET_LAYERS)}")
    log.info(f"Started at    : {datetime.now().isoformat()}")

    all_configs = []

    for rank in RANKS:
        for layer_idx in TARGET_LAYERS:
            log.info("=" * 60)
            log.info(f"Training SAE | rank={rank} | layer={layer_idx}")
            log.info("=" * 60)

            # Skip if already trained
            out_dir      = os.path.join(SAE_ROOT, f"r{rank}", f"layer_{layer_idx}")
            weights_path = os.path.join(out_dir, "sae_weights.pt")
            if os.path.exists(weights_path):
                log.info(f"  Already trained. Skipping.")
                continue

            try:
                # Load delta vectors
                delta_vectors, rms_scale = load_delta_vectors(rank, layer_idx)

                # Train
                sae, train_log = train_sae(rank, layer_idx, delta_vectors)

                # Save immediately
                config = save_sae(rank, layer_idx, sae, train_log, rms_scale)
                all_configs.append(config)

                # Cleanup
                del sae, delta_vectors
                gc.collect()
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                elif DEVICE == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                log.error(f"  FAILED rank={rank} layer={layer_idx}: {e}", exc_info=True)
                all_configs.append({
                    "rank": rank, "layer": layer_idx,
                    "status": "FAILED", "error": str(e)
                })
            finally:
                # Always update summary
                summary_path = os.path.join(SAE_ROOT, "training_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(all_configs, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("ALL SAEs COMPLETE — Final Summary")
    log.info("=" * 60)
    for cfg in all_configs:
        if "error" in cfg:
            log.info(f"  r={cfg['rank']} | layer={cfg['layer']} | FAILED: {cfg['error']}")
        else:
            log.info(
                f"  r={cfg['rank']:>2} | layer={cfg['layer']:>2} | "
                f"l0={cfg['final_avg_l0']:>6.1f} | "
                f"mse={cfg['final_avg_mse']:.6f}"
            )

    log.info(f"\nSAEs saved in: {os.path.abspath(SAE_ROOT)}/")
    log.info("Next step: comparative analysis — delta SAEs vs Gemma Scope SAEs.")


if __name__ == "__main__":
    main()
