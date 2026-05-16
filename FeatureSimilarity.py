"""
Dictionary Similarity Analysis
LoRA × Mechanistic Interpretability Research
=============================================
Computes cosine similarity between delta SAE decoder directions
and Gemma Scope (base model) SAE decoder directions.

For each delta SAE feature direction d_i, finds the maximum cosine
similarity to any Gemma Scope feature direction g_j.

Interpretation:
    max_sim > 0.7   → feature shared with base model (amplification)
    max_sim 0.3–0.7 → partial alignment (related but distinct)
    max_sim < 0.3   → genuinely novel direction (adapter-specific)

"""

import os
import gc
import json
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from sae_lens import SAE

# ── Dirs ──────────────────────────────────────────────────────────────────────
RESULTS_ROOT  = "./dict_similarity"
DELTA_SAE_ROOT = "./delta_saes"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_ROOT, "similarity.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANKS         = [4, 8, 16, 32]
TARGET_LAYERS = [5, 10, 18, 22, 32, 38]
SAE_IDS       = {
    5 : "layer_5/width_16k/average_l0_20",
    10: "layer_10/width_16k/average_l0_17",
    18: "layer_18/width_16k/average_l0_20",
    22: "layer_22/width_16k/average_l0_19",
    32: "layer_32/width_16k/average_l0_20",
    38: "layer_38/width_16k/average_l0_20",
}
SAE_RELEASE = "gemma-scope-9b-pt-res"

# Similarity thresholds
THRESH_NOVEL   = 0.3   # below = novel/adapter-specific
THRESH_SHARED  = 0.7   # above = shared with base model
CHUNK_SIZE     = 512   # rows of delta SAE to process at once (memory management)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
log.info(f"Device: {DEVICE}")


# ── Load Gemma Scope decoder directions ───────────────────────────────────────
def load_gemma_scope_decoder(layer_idx):
    log.info(f"Loading Gemma Scope SAE for layer {layer_idx}...")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_IDS[layer_idx],
    )
    # W_dec: [d_sae, d_model] — each row is one feature direction
    W_dec = sae.W_dec.detach().to(torch.float32)

    # Normalise rows to unit vectors for cosine similarity
    W_dec = F.normalize(W_dec, dim=1)
    log.info(f"Gemma Scope decoder shape: {W_dec.shape}")
    del sae
    gc.collect()
    return W_dec


# ── Load delta SAE decoder directions ────────────────────────────────────────
def load_delta_sae_decoder(rank, layer_idx):
    weights_path = os.path.join(DELTA_SAE_ROOT, f"r{rank}", f"layer_{layer_idx}", "sae_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Delta SAE not found: {weights_path}")

    weights = torch.load(weights_path, map_location="cpu")
    W_dec   = weights["W_dec"].to(torch.float32)   # [d_sae, d_model]

    # Normalise rows to unit vectors
    W_dec = F.normalize(W_dec, dim=1)
    log.info(f"Delta SAE decoder shape: {W_dec.shape}")
    return W_dec


# ── Chunked max cosine similarity ─────────────────────────────────────────────
def compute_max_cosine_similarity(delta_W_dec, base_W_dec):
    """
    For each delta feature direction, finds:
        - max cosine similarity to any base feature
        - index of the best matching base feature

    Computed in chunks to manage memory.

    Args:
        delta_W_dec: [d_sae_delta, d_model] unit vectors
        base_W_dec:  [d_sae_base,  d_model] unit vectors

    Returns:
        max_sims:    [d_sae_delta] max similarity per delta feature
        best_idxs:   [d_sae_delta] best matching base feature index
    """
    n_delta = delta_W_dec.shape[0]
    max_sims  = torch.zeros(n_delta)
    best_idxs = torch.zeros(n_delta, dtype=torch.long)

    # Move base decoder to device once
    base_W_dec_dev = base_W_dec.to(DEVICE)

    for start in range(0, n_delta, CHUNK_SIZE):
        end   = min(start + CHUNK_SIZE, n_delta)
        chunk = delta_W_dec[start:end].to(DEVICE)   # [chunk, d_model]

        # Cosine similarity matrix: [chunk, d_sae_base]
        # Since both are unit normalised: sim = chunk @ base.T
        sim_matrix = chunk @ base_W_dec_dev.T        # [chunk, d_sae_base]

        chunk_max_sims, chunk_best_idxs = sim_matrix.max(dim=1)

        max_sims[start:end]  = chunk_max_sims.cpu()
        best_idxs[start:end] = chunk_best_idxs.cpu()

        del chunk, sim_matrix
        if DEVICE == "mps":
            torch.mps.empty_cache()

        if (start // CHUNK_SIZE) % 10 == 0:
            log.info(f"  Processed {end}/{n_delta} delta features...")

    return max_sims, best_idxs


# ── Build similarity histogram ────────────────────────────────────────────────
def build_histogram(max_sims, n_bins=20):
    bins   = torch.linspace(0, 1, n_bins + 1)
    counts = torch.histc(max_sims, bins=n_bins, min=0.0, max=1.0)
    return {
        "bin_edges" : [round(b, 3) for b in bins.tolist()],
        "counts"    : counts.long().tolist(),
    }


# ── Compute similarity stats for one (rank, layer) ───────────────────────────
def compute_similarity(rank, layer_idx, delta_W_dec, base_W_dec):
    log.info(f"  Computing max cosine similarity [{delta_W_dec.shape[0]} × {base_W_dec.shape[0]}]...")
    max_sims, best_idxs = compute_max_cosine_similarity(delta_W_dec, base_W_dec)

    # ── Summary statistics ────────────────────────────────────────────────────
    n_total   = max_sims.shape[0]
    n_novel   = (max_sims < THRESH_NOVEL).sum().item()
    n_partial = ((max_sims >= THRESH_NOVEL) & (max_sims < THRESH_SHARED)).sum().item()
    n_shared  = (max_sims >= THRESH_SHARED).sum().item()

    pct_novel   = round(100 * n_novel   / n_total, 2)
    pct_partial = round(100 * n_partial / n_total, 2)
    pct_shared  = round(100 * n_shared  / n_total, 2)

    mean_sim   = round(max_sims.mean().item(), 6)
    median_sim = round(max_sims.median().item(), 6)
    std_sim    = round(max_sims.std().item(), 6)

    log.info(f"  Mean max similarity  : {mean_sim:.4f}")
    log.info(f"  Median max similarity: {median_sim:.4f}")
    log.info(f"  Novel   (<{THRESH_NOVEL}): {pct_novel}%  ({n_novel}/{n_total})")
    log.info(f"  Partial ({THRESH_NOVEL}–{THRESH_SHARED}): {pct_partial}% ({n_partial}/{n_total})")
    log.info(f"  Shared  (>{THRESH_SHARED}): {pct_shared}%  ({n_shared}/{n_total})")

    # ── Top most similar delta features (closest to base) ────────────────────
    top_k = 20
    top_vals, top_delta_idxs = torch.topk(max_sims, top_k)
    top_base_idxs = best_idxs[top_delta_idxs]

    # ── Top most novel delta features (furthest from base) ───────────────────
    bot_vals, bot_delta_idxs = torch.topk(max_sims, top_k, largest=False)
    bot_base_idxs = best_idxs[bot_delta_idxs]

    # ── Histogram ─────────────────────────────────────────────────────────────
    histogram = build_histogram(max_sims)

    # ── Package results ───────────────────────────────────────────────────────
    results = {
        "rank"        : rank,
        "layer"       : layer_idx,
        "n_delta_feats": n_total,
        "n_base_feats" : base_W_dec.shape[0],

        # Summary stats
        "mean_max_sim"  : mean_sim,
        "median_max_sim": median_sim,
        "std_max_sim"   : std_sim,

        # Threshold breakdown
        "thresh_novel"  : THRESH_NOVEL,
        "thresh_shared" : THRESH_SHARED,
        "pct_novel"     : pct_novel,
        "pct_partial"   : pct_partial,
        "pct_shared"    : pct_shared,
        "n_novel"       : n_novel,
        "n_partial"     : n_partial,
        "n_shared"      : n_shared,

        # Top most similar (closest to base model features)
        "top_similar": {
            "delta_feature_idxs": top_delta_idxs.tolist(),
            "base_feature_idxs" : top_base_idxs.tolist(),
            "similarity_values" : [round(v, 6) for v in top_vals.tolist()],
        },

        # Top most novel (furthest from base model features)
        "top_novel": {
            "delta_feature_idxs": bot_delta_idxs.tolist(),
            "base_feature_idxs" : bot_base_idxs.tolist(),
            "similarity_values" : [round(v, 6) for v in bot_vals.tolist()],
        },

        # Full histogram for plotting
        "histogram": histogram,

        "computed_at": datetime.now().isoformat(),
    }

    return results, max_sims


# ── Process one rank ──────────────────────────────────────────────────────────
def process_rank(rank, base_decoders):
    log.info("=" * 60)
    log.info(f"Processing rank={rank}")
    log.info("=" * 60)

    rank_res_dir = os.path.join(RESULTS_ROOT, f"r{rank}")
    os.makedirs(rank_res_dir, exist_ok=True)
    rank_results = []

    for layer_idx in TARGET_LAYERS:
        log.info(f"\n[r={rank}] Layer {layer_idx}")

        # Skip if already done
        out_path = os.path.join(rank_res_dir, f"layer_{layer_idx}_similarity.json")
        if os.path.exists(out_path):
            log.info(f"  Already computed. Skipping.")
            with open(out_path) as f:
                rank_results.append(json.load(f))
            continue

        try:
            # Load delta SAE decoder
            delta_W_dec = load_delta_sae_decoder(rank, layer_idx)

            # Get base decoder (cached)
            base_W_dec = base_decoders[layer_idx]

            # Compute similarity
            results, max_sims = compute_similarity(rank, layer_idx, delta_W_dec, base_W_dec)

            # Save immediately
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"  ✅ Saved: {out_path}")

            rank_results.append(results)

            del delta_W_dec, max_sims
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

        except Exception as e:
            log.error(f"  FAILED r={rank} layer={layer_idx}: {e}", exc_info=True)

    return rank_results


# ── Build cross-rank summary ──────────────────────────────────────────────────
def build_summary(all_results):
    summary = {}
    for rank, rank_results in all_results.items():
        summary[f"r{rank}"] = {}
        for res in rank_results:
            layer = res["layer"]
            summary[f"r{rank}"][f"layer_{layer}"] = {
                "mean_max_sim"  : res["mean_max_sim"],
                "median_max_sim": res["median_max_sim"],
                "pct_novel"     : res["pct_novel"],
                "pct_partial"   : res["pct_partial"],
                "pct_shared"    : res["pct_shared"],
            }
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   Dictionary Similarity — LoRA × Mech Interp Research   ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Ranks         : {RANKS}")
    log.info(f"Layers        : {TARGET_LAYERS}")
    log.info(f"Device        : {DEVICE}")
    log.info(f"Thresholds    : novel<{THRESH_NOVEL} | shared>{THRESH_SHARED}")
    log.info(f"Started at    : {datetime.now().isoformat()}")

    # Load all Gemma Scope decoders once — reused across all ranks
    log.info("\nLoading Gemma Scope decoders for all layers...")
    base_decoders = {}
    for layer_idx in TARGET_LAYERS:
        base_decoders[layer_idx] = load_gemma_scope_decoder(layer_idx)
    log.info("All Gemma Scope decoders loaded.\n")

    all_results = {}

    for rank in RANKS:
        try:
            rank_results = process_rank(rank, base_decoders)
            all_results[rank] = rank_results
        except Exception as e:
            log.error(f"[r={rank}] FAILED: {e}", exc_info=True)
            all_results[rank] = []
        finally:
            summary = build_summary(all_results)
            summary_path = os.path.join(RESULTS_ROOT, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            log.info(f"Summary updated: {summary_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("ALL RANKS COMPLETE — Dictionary Similarity Summary")
    log.info("=" * 60)

    summary = build_summary(all_results)
    for rank_key, layers in summary.items():
        log.info(f"\n{rank_key}:")
        for layer_key, metrics in layers.items():
            log.info(
                f"  {layer_key} | "
                f"mean_sim={metrics['mean_max_sim']:.4f} | "
                f"novel={metrics['pct_novel']}% | "
                f"partial={metrics['pct_partial']}% | "
                f"shared={metrics['pct_shared']}%"
            )

    log.info(f"\nResults saved in: {os.path.abspath(RESULTS_ROOT)}/")
    log.info("Next step: visualise similarity distributions and interpret top features.")


if __name__ == "__main__":
    main()
