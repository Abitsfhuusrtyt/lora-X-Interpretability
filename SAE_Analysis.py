"""
SAE Analysis Script for LoRA × Mechanistic Interpretability Research
======================================================================
Loads saved h_base and h_delta activations, passes them through
Gemma Scope SAEs, and records:
  - Active features and their magnitudes
  - Reconstruction error (how much delta the SAE cannot explain)
  - Feature overlap between delta and base activations
  - Feature activation statistics across ranks and layers

"""

import os
import gc
import json
import logging
import torch
import numpy as np
from datetime import datetime
from sae_lens import SAE

# ── Output dirs ───────────────────────────────────────────────────────────────
RESULTS_ROOT    = "./sae_results"
ACTIVATIONS_ROOT = "./activations_v2"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_ROOT, "sae_analysis.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANKS      = [4, 8, 16, 32]
SAE_IDS    = {
    5 : "layer_5/width_16k/average_l0_20",
    10: "layer_10/width_16k/average_l0_17",
    18: "layer_18/width_16k/average_l0_20",
    22: "layer_22/width_16k/average_l0_19",
    32: "layer_32/width_16k/average_l0_20",
    38: "layer_38/width_16k/average_l0_20",
}
SAE_RELEASE     = "gemma-scope-9b-pt-res"
FEATURE_ACT_THRESHOLD = 0.01   # minimum activation magnitude to count as "active"
TOP_K_FEATURES        = 20     # top K features to store per sample per layer

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
log.info(f"Device: {DEVICE}")


# ── Load SAE for a given layer ────────────────────────────────────────────────
def load_sae(layer_idx):
    sae_id = SAE_IDS[layer_idx]
    log.info(f"Loading SAE: {SAE_RELEASE} / {sae_id}")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
    )
    sae = sae.to(torch.float32)
    sae.eval()
    log.info(f"SAE loaded. d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    return sae


# ── Run SAE on a batch of activations ────────────────────────────────────────
@torch.no_grad()
def run_sae(sae, activations, batch_size=8):
    # Handle both [n_samples, seq_len, d_model] and [N, d_model]
    if activations.dim() == 3:
        n_samples, seq_len, d_model = activations.shape
        flat = activations.reshape(n_samples * seq_len, d_model)
    else:
        flat = activations  # already [N, d_model]

    N = flat.shape[0]
    all_feature_acts  = []
    all_reconstructed = []

    for start in range(0, N, batch_size * 512):
        end   = min(start + batch_size * 512, N)
        batch = flat[start:end].to(torch.float32)

        feature_acts  = sae.encode(batch)
        reconstructed = sae.decode(feature_acts)

        all_feature_acts.append(feature_acts.cpu())
        all_reconstructed.append(reconstructed.cpu())

    feature_acts  = torch.cat(all_feature_acts, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)
    error         = flat.to(torch.float32).cpu() - reconstructed

    return feature_acts, reconstructed, error


# ── Compute metrics for one layer ─────────────────────────────────────────────
def compute_metrics(
    layer_idx,
    h_base,       # [n_samples, seq_len, d_model]
    h_delta,      # [n_samples, seq_len, d_model]
    sae,
):
    """
    Core analysis:
    1. Run SAE on h_base  → base feature activations
    2. Run SAE on h_delta → delta feature activations
    3. Compute reconstruction error on delta
    4. Compute feature overlap between base and delta
    5. Compute feature density (how many features active per token)
    """
    log.info(f"  Running SAE on h_base  (shape: {h_base.shape})...")
    base_feats, base_recon, base_error = run_sae(sae, h_base)

    log.info(f"  Running SAE on h_delta (shape: {h_delta.shape})...")
    delta_feats, delta_recon, delta_error = run_sae(sae, h_delta)

    # ── Reconstruction error ──────────────────────────────────────────────────
    # Relative reconstruction error = ||error|| / ||input||
    delta_norm       = h_delta.to(torch.float32).norm(dim=-1)          # [n, s]
    delta_error_norm = delta_error.norm(dim=-1)                         # [n, s]

    # Avoid division by zero for padding tokens
    relative_error = torch.where(
        delta_norm > 1e-6,
        delta_error_norm / delta_norm,
        torch.zeros_like(delta_norm)
    )

    mean_relative_error = relative_error.mean().item()
    mean_delta_norm     = delta_norm.mean().item()

    log.info(f"  Mean delta norm         : {mean_delta_norm:.6f}")
    log.info(f"  Mean relative recon error: {mean_relative_error:.6f}")

    # ── Feature density ───────────────────────────────────────────────────────
    # How many features activate per token (above threshold)
    base_active_mask  = (base_feats  > FEATURE_ACT_THRESHOLD)  # [n, s, d_sae]
    delta_active_mask = (delta_feats > FEATURE_ACT_THRESHOLD)  # [n, s, d_sae]

    base_density  = base_active_mask.float().sum(dim=-1).mean().item()
    delta_density = delta_active_mask.float().sum(dim=-1).mean().item()

    log.info(f"  Mean active features (base) : {base_density:.2f}")
    log.info(f"  Mean active features (delta): {delta_density:.2f}")
    # ── Feature overlap ───────────────────────────────────────────────────────
    # For each token, what fraction of delta-active features
    # are also active in h_base? (amplification vs dormant vs new)
    both_active        = (base_active_mask & delta_active_mask).float().sum(dim=-1)
    delta_only         = (~base_active_mask & delta_active_mask).float().sum(dim=-1)
    total_delta_active = delta_active_mask.float().sum(dim=-1).clamp(min=1e-6)
    overlap_fraction   = (both_active / total_delta_active).mean().item()
    novel_fraction     = (delta_only  / total_delta_active).mean().item()

    log.info(f"  Feature overlap (amplified): {overlap_fraction:.4f}")
    log.info(f"  Novel delta features       : {novel_fraction:.4f}")

    # ── Top features ──────────────────────────────────────────────────────────
    # Most consistently active features across samples (mean pooled over seq)
    # Shape: [n_samples, d_sae] after mean over seq
    delta_feats_mean = delta_feats.mean(dim=1)  # [n, d_sae]
    base_feats_mean  = base_feats.mean(dim=1)   # [n, d_sae]

    # Mean activation across all samples
    delta_feats_global = delta_feats.mean(dim=0)   # [d_sae]
    base_feats_global  = base_feats.mean(dim=0)     # [d_sae]

    # Top-K most active features in delta
    k_delta = max(1, min(TOP_K_FEATURES, (delta_feats_global > 0).sum().item()))
    k_base  = max(1, min(TOP_K_FEATURES, (base_feats_global  > 0).sum().item()))

    top_delta_vals, top_delta_idxs = torch.topk(delta_feats_global, k_delta)
    top_base_vals,  top_base_idxs  = torch.topk(base_feats_global,  k_base)

# Ensure always a list, even for k=1
    top_delta_set = set(top_delta_idxs.reshape(-1).tolist())
    top_base_set  = set(top_base_idxs.reshape(-1).tolist())
    top_k_overlap = len(top_delta_set & top_base_set)

    log.info(f"  Top-{TOP_K_FEATURES} feature overlap: {top_k_overlap}/{TOP_K_FEATURES}")

    # ── Per-sample feature consistency ────────────────────────────────────────
    # How consistent are the active features across samples?
    # High consistency = the adapter reliably activates the same features
    delta_active_per_sample = delta_active_mask.float().mean(dim=1)  # [n, d_sae]
    feature_consistency = delta_active_mask.float().mean(dim=0)   # [d_sae]
    mean_consistency    = feature_consistency.mean().item()

    log.info(f"  Mean feature consistency: {mean_consistency:.4f}")

    # ── Package results ───────────────────────────────────────────────────────
    results = {
        "layer"                    : layer_idx,
        "n_samples" : h_delta.shape[0],
        "seq_len"   : 1,                  # flattened — no seq dimension
        "d_model"   : h_delta.shape[1],
        "d_sae"                    : sae.cfg.d_sae,
        "sae_id"                   : SAE_IDS[layer_idx],

        # Delta magnitude
        "mean_delta_norm"          : round(mean_delta_norm, 6),

        # Reconstruction error
        "mean_relative_recon_error": round(mean_relative_error, 6),

        # Feature density
        "mean_active_features_base" : round(base_density, 4),
        "mean_active_features_delta": round(delta_density, 4),

        # Feature overlap (amplified vs novel)
        "overlap_fraction"         : round(overlap_fraction, 4),
        "novel_fraction"           : round(novel_fraction, 4),

        # Top-K features
        "top_k"                    : TOP_K_FEATURES,
        "top_delta_feature_idxs"   : top_delta_idxs.tolist(),
        "top_delta_feature_vals"   : [round(v, 6) for v in top_delta_vals.tolist()],
        "top_base_feature_idxs"    : top_base_idxs.tolist(),
        "top_base_feature_vals"    : [round(v, 6) for v in top_base_vals.tolist()],
        "top_k_overlap_count"      : top_k_overlap,

        # Consistency
        "mean_feature_consistency" : round(mean_consistency, 6),

        "computed_at"              : datetime.now().isoformat(),
    }

    return results


# ── Process one rank ──────────────────────────────────────────────────────────
def process_rank(rank):
    log.info("=" * 60)
    log.info(f"Processing rank={rank}")
    log.info("=" * 60)

    rank_act_dir = os.path.join(ACTIVATIONS_ROOT, f"r{rank}")
    rank_res_dir = os.path.join(RESULTS_ROOT, f"r{rank}")
    os.makedirs(rank_res_dir, exist_ok=True)

    base_dir     = os.path.join(ACTIVATIONS_ROOT, "base")
    rank_results = []

    for layer_idx, sae_id in SAE_IDS.items():
        log.info(f"\n[r={rank}] Layer {layer_idx}")

        # ── Load activations ──────────────────────────────────────────────────
        h_base_path  = os.path.join(base_dir,     f"layer_{layer_idx}_h_base.pt")
        h_delta_path = os.path.join(rank_act_dir, f"layer_{layer_idx}_h_delta.pt")

        if not os.path.exists(h_base_path):
            log.error(f"h_base not found: {h_base_path}")
            continue
        if not os.path.exists(h_delta_path):
            log.error(f"h_delta not found: {h_delta_path}")
            continue

        log.info(f"[r={rank}] Loading h_base  from {h_base_path}...")
        h_base  = torch.load(h_base_path,  map_location="cpu")
        log.info(f"[r={rank}] Loading h_delta from {h_delta_path}...")
        h_delta = torch.load(h_delta_path, map_location="cpu")

        log.info(f"[r={rank}] h_base  shape: {h_base.shape}")
        log.info(f"[r={rank}] h_delta shape: {h_delta.shape}")

        # ── Load SAE ──────────────────────────────────────────────────────────
        sae = load_sae(layer_idx)

        # ── Compute metrics ───────────────────────────────────────────────────
        results = compute_metrics(layer_idx, h_base, h_delta, sae)
        results["rank"] = rank

        # ── Save layer results immediately ────────────────────────────────────
        out_path = os.path.join(rank_res_dir, f"layer_{layer_idx}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"[r={rank}] ✅ Layer {layer_idx} results saved to {out_path}")

        rank_results.append(results)

        # Cleanup
        del h_base, h_delta, sae
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    return rank_results


# ── Build cross-rank summary ──────────────────────────────────────────────────
def build_summary(all_results):
    """
    Builds a summary table for easy comparison across ranks and layers.
    Key metrics per rank per layer:
      - mean_delta_norm
      - mean_relative_recon_error
      - mean_active_features_delta
      - overlap_fraction
      - novel_fraction
    """
    summary = {}

    for rank, rank_results in all_results.items():
        summary[f"r{rank}"] = {}
        for res in rank_results:
            layer = res["layer"]
            summary[f"r{rank}"][f"layer_{layer}"] = {
                "mean_delta_norm"          : res["mean_delta_norm"],
                "mean_relative_recon_error": res["mean_relative_recon_error"],
                "mean_active_features_delta": res["mean_active_features_delta"],
                "overlap_fraction"         : res["overlap_fraction"],
                "novel_fraction"           : res["novel_fraction"],
                "top_k_overlap_count"      : res["top_k_overlap_count"],
                "mean_feature_consistency" : res["mean_feature_consistency"],
            }

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║       SAE Analysis — LoRA × Mech Interp Research        ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"SAE release    : {SAE_RELEASE}")
    log.info(f"Ranks          : {RANKS}")
    log.info(f"Layers         : {list(SAE_IDS.keys())}")
    log.info(f"Results root   : {RESULTS_ROOT}")
    log.info(f"Started at     : {datetime.now().isoformat()}")

    all_results = {}

    for rank in RANKS:
        try:
            rank_results = process_rank(rank)
            all_results[rank] = rank_results
        except Exception as e:
            log.error(f"[r={rank}] FAILED: {e}", exc_info=True)
            all_results[rank] = []
        finally:
            # Save summary after every rank
            summary = build_summary(all_results)
            summary_path = os.path.join(RESULTS_ROOT, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            log.info(f"Summary updated: {summary_path}")

    # ── Final summary print ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("ALL RANKS COMPLETE — Key Metrics Summary")
    log.info("=" * 60)

    summary = build_summary(all_results)
    for rank_key, layers in summary.items():
        log.info(f"\n{rank_key}:")
        for layer_key, metrics in layers.items():
            log.info(
                f"  {layer_key} | "
                f"delta_norm={metrics['mean_delta_norm']:.4f} | "
                f"recon_err={metrics['mean_relative_recon_error']:.4f} | "
                f"active_feats={metrics['mean_active_features_delta']:.2f} | "
                f"overlap={metrics['overlap_fraction']:.4f} | "
                f"novel={metrics['novel_fraction']:.4f}"
            )

    log.info(f"\nAll results saved in: {os.path.abspath(RESULTS_ROOT)}/")
    log.info("Next step: plot and interpret the summary metrics.")


if __name__ == "__main__":
    main()
