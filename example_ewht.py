from __future__ import annotations
from math import comb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from ewht import (
    efwht_from_dataframe,
    example_msa,
    genotypes_from_dataframe,
    get_ps,
    load_example,
    run_cs_experiment,
    sample_evolutionary_sequences,
)

OUTPUT_DIR = Path("example_ewht")
WT_SEQUENCE = (
    "QVQLVESGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGPEWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAMYYCAKHMGYQLRETMDVWGQGTTVTVSS"
)
POSITIONS = [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104]
MUTANTS = ["P", "R", "T", "K", "P", "D", "F", "A", "G", "V", "V"]
TRAIN_N = 100
MIN_ORDER = 1
MAX_ORDER = 5

ORDER_COLORS = {
    1: "#ef2b25",
    2: "#f59720",
    3: "#1cb04b",
    4: "#5466ac",
    5: "#ad2b8f",
}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data and preprocess
    raw = load_example()
    print(raw.head())
    df = genotypes_from_dataframe(
        raw,
        POSITIONS,
        wt_sequence=WT_SEQUENCE,
        mutants=MUTANTS,
        sequence_col="mutated_sequence",
        fitness_col="fitness",
    )
    L = len(POSITIONS)
    print(df.head())
    print(f"{df['genotype'].nunique()} unique genotypes, L={L}")

    with example_msa() as msa_path:
        # Compute ps from MSA
        ps = get_ps(WT_SEQUENCE, POSITIONS, MUTANTS, msa=msa_path)
        plot_ps(ps, OUTPUT_DIR / "ps_from_msa.png")

        # Compute eWHT
        coeffs, center = efwht_from_dataframe(df, ps, basis="eWHT")
        plot_ewht_spectrum(coeffs, L, OUTPUT_DIR / "ewht_spectrum.png", max_order=MAX_ORDER)

        # Sample evolutionary sequences for compressed sensing
        train, test = sample_evolutionary_sequences(
            df,
            ps,
            msa=msa_path,
            positions=POSITIONS,
            wt_sequence=WT_SEQUENCE,
            mutants=MUTANTS,
            fraction=0.75,
            train_n=TRAIN_N,
            random_state=0,
        )
        print(f"train={len(train)}, test={len(test)}")

        # Run compressed sensing experiment
        result = run_cs_experiment(train, test, ps, basis="eWHT", center_by_ps=True, random_state=0)
        print(f"best lambda: {result.best_lambda}")
        print(f"train R²: {result.train_metrics['r2']:.4f}")
        print(f"test R²:  {result.test_metrics['r2']:.4f}")

    print(f"Figures in {OUTPUT_DIR.resolve()}/")


"""
Utility functions for plotting
"""
def interaction_order_array(L: int) -> np.ndarray:
    return np.array([i.bit_count() for i in range(2**L)], dtype=int)


def order_boundaries(L: int, min_order: int, max_order: int) -> list[int]:
    boundaries: list[int] = []
    cum = 0
    for k in range(min_order, max_order + 1):
        cum += comb(L, k)
        if k < max_order:
            boundaries.append(cum)
    return boundaries


def normalize_coeff_magnitudes(coeffs: np.ndarray) -> np.ndarray:
    denom = float(np.sqrt(np.sum(np.square(coeffs))))
    return coeffs / denom if denom > 0 else np.zeros_like(coeffs)


def _order_label(order: int) -> str:
    if order == 1:
        return "1st"
    if order == 2:
        return "2nd"
    if order == 3:
        return "3rd"
    return f"{order}th"


def plot_ps(ps: np.ndarray, out_path: Path) -> None:
    labels = [f"{pos}\n({mut})" for pos, mut in zip(POSITIONS, MUTANTS)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ps)), ps, color="darkorange", edgecolor="white")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Position (mutant amino acid)")
    ax.set_ylabel("p(mutant)")
    ax.set_title("Evolutionary mutation probabilities from bundled MSA")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="p = 0.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ewht_spectrum(
    coeffs: np.ndarray,
    L: int,
    out_path: Path,
    *,
    min_order: int = MIN_ORDER,
    max_order: int = MAX_ORDER,
) -> None:
    orders = interaction_order_array(L)
    perm = np.argsort(orders, kind="stable")
    mags = np.abs(normalize_coeff_magnitudes(coeffs)[perm])
    ord_sorted = orders[perm]

    start = sum(comb(L, k) for k in range(0, min_order))
    end = sum(comb(L, k) for k in range(0, max_order + 1))
    mags = mags[start:end]
    ord_sorted = ord_sorted[start:end]
    x = np.arange(len(mags))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for order in range(min_order, max_order + 1):
        mask = ord_sorted == order
        if not np.any(mask):
            continue
        color = ORDER_COLORS.get(order, "#b3b3b3")
        _, stemlines, _ = ax.stem(x[mask], mags[mask], linefmt=color, markerfmt=" ", basefmt=" ")
        plt.setp(stemlines, linewidth=0.8)

    for boundary in order_boundaries(L, min_order, max_order):
        if 0 < boundary < len(mags):
            ax.axvline(boundary, color="gray", linestyle="--", alpha=0.35, linewidth=0.8)

    ax.set_xlim(-0.5, len(mags) - 0.5)
    ax.set_ylabel("|coefficient| (normalized)")
    ax.set_xlabel("Coefficient index (sorted by interaction order)")
    ax.set_title(f"eWHT spectrum (orders {min_order}–{max_order})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=ORDER_COLORS[o], lw=1.2, label=_order_label(o))
        for o in range(min_order, max_order + 1)
        if o in ORDER_COLORS
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
