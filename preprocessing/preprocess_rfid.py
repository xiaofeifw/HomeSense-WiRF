import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def moving_average(values, window_size=3):
    """
    Simple moving average denoising (optional).
    Note: This function is not used by default.
    """
    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")


def denoise_phase_by_epc(
    df: pd.DataFrame,
    epc_col: str = "EPC",
    phase_col: str = "PhaseAngle(Radian)",
    sigma_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Denoise RFID phase by detecting outliers within each EPC group and
    replacing them using linear interpolation over time order.

    Outlier rule (per EPC):
      phase > mean + sigma_factor * std  OR  phase < mean - sigma_factor * std

    Returns:
      A new DataFrame with denoised phase values.
    """
    df = df.copy()

    # Per-EPC statistics aligned to each row
    grouped_mean = df.groupby(epc_col)[phase_col].transform("mean")
    grouped_std = df.groupby(epc_col)[phase_col].transform("std")

    upper = grouped_mean + sigma_factor * grouped_std
    lower = grouped_mean - sigma_factor * grouped_std

    # Process each EPC independently
    for epc in df[epc_col].unique():
        epc_mask = (df[epc_col] == epc)
        epc_idx = df.index[epc_mask]
        epc_phase = df.loc[epc_idx, phase_col]

        # Indices of outliers
        outlier_idx = epc_idx[(epc_phase > upper.loc[epc_idx]) | (epc_phase < lower.loc[epc_idx])]

        if len(outlier_idx) == 0:
            continue

        # Use remaining points to interpolate outliers
        x_all = epc_idx.values.astype(float)
        y_all = epc_phase.values.astype(float)

        keep_mask = ~np.isin(epc_idx.values, outlier_idx.values)
        if keep_mask.sum() < 2:
            # Not enough points to interpolate
            continue

        f = interp1d(
            x_all[keep_mask],
            y_all[keep_mask],
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )

        df.loc[outlier_idx, phase_col] = f(outlier_idx.values.astype(float))

    return df


def plot_phase_by_epc(
    df: pd.DataFrame,
    epc_col: str = "EPC",
    phase_col: str = "PhaseAngle(Radian)",
    save_path: str = None,
):
    """
    Optional visualization: plot phase traces grouped by EPC.
    If save_path is provided, save the figure to disk (no interactive display).
    """
    ax = None
    for _, group in df.groupby(epc_col):
        ax = group[phase_col].plot(ax=ax, legend=False)

    plt.axis("off")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0, dpi=200)

    plt.close()


def process_single_csv(
    csv_path: str,
    target_frame: int = 148,
    epc_col: str = "EPC",
    time_col: str = "UNIX_time",
    phase_col: str = "PhaseAngle(Radian)",
    expected_max_tags: int = 24,
    stats: dict = None,
    denoise: bool = True,
    sigma_factor: float = 1.5,
    save_plot: bool = False,
    plot_dir: str = None,
):
    """
    Process one RFID CSV file:
      1) Sort by time_col
      2) (Optional) Denoise phase per EPC by outlier interpolation
      3) For each EPC group, resample to target_frame using linear interpolation
      4) Pad/truncate tags to expected_max_tags

    Returns:
      tag_features: numpy array with shape (expected_max_tags, target_frame)
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(time_col)

    if denoise:
        df = denoise_phase_by_epc(
            df,
            epc_col=epc_col,
            phase_col=phase_col,
            sigma_factor=sigma_factor,
        )

    if save_plot and plot_dir is not None:
        plot_path = os.path.join(plot_dir, os.path.basename(csv_path).replace(".csv", ".png"))
        plot_phase_by_epc(df, epc_col=epc_col, phase_col=phase_col, save_path=plot_path)

    tag_groups = df.groupby(epc_col)
    tag_features = []

    print(f"[INFO] Processing: {csv_path}")
    print(f"[INFO] Number of tags: {len(tag_groups)}")

    for tag_id, group in tag_groups:
        values = group[phase_col].values.astype(float)
        count = len(values)
        print(f"  - Tag {tag_id} has {count} frames")

        if stats is not None:
            stats[tag_id].append(count)

        if count < 2:
            continue

        resampled = np.interp(
            np.linspace(0, count - 1, target_frame),
            np.arange(count),
            values
        )
        tag_features.append(resampled)

    tag_features = np.array(tag_features, dtype=float)

    # Pad or truncate to expected_max_tags
    if tag_features.shape[0] < expected_max_tags:
        pad = np.zeros((expected_max_tags - tag_features.shape[0], target_frame), dtype=float)
        tag_features = np.vstack([tag_features, pad])
    else:
        tag_features = tag_features[:expected_max_tags]

    return tag_features


def batch_convert_rfid_csv(
    csv_dir: str,
    output_dir: str,
    subject_id: str,
    action_id: str,
    repeats_per_action: int = 20,
    target_frame: int = 148,
    expected_max_tags: int = 24,
    denoise: bool = True,
    sigma_factor: float = 1.5,
    save_plot: bool = False,
    plot_dir: str = None,
):
    """
    Batch convert RFID CSV files to .npy files.

    Output naming:
      <subject>_<action>_<repeat:02d>.npy

    NOTE:
      This function assumes the input directory contains exactly `repeats_per_action` CSV files
      for one (subject_id, action_id) pair. If your directory contains multiple actions, you can
      call this function per action folder, or extend the naming logic accordingly.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")])
    assert len(csv_files) % repeats_per_action == 0, \
        f"Expected multiples of {repeats_per_action} repeats, but got {len(csv_files)} files."

    stats = defaultdict(list)

    for idx, f in enumerate(tqdm(csv_files, desc="Converting")):
        csv_path = os.path.join(csv_dir, f)

        feature_array = process_single_csv(
            csv_path,
            target_frame=target_frame,
            expected_max_tags=expected_max_tags,
            stats=stats,
            denoise=denoise,
            sigma_factor=sigma_factor,
            save_plot=save_plot,
            plot_dir=plot_dir,
        )

        repeat_num = idx % repeats_per_action + 1
        npy_name = f"{subject_id}_{action_id}_{repeat_num:02d}.npy"
        np.save(os.path.join(output_dir, npy_name), feature_array)

    print(f"[DONE] Processed {len(csv_files)} files. Output saved to: {output_dir}")

    print("\n[STATS] Original frame count per tag (EPC):")
    for tag_id, counts in stats.items():
        print(f"Tag {tag_id} - min: {min(counts)}, max: {max(counts)}, avg: {sum(counts) / len(counts):.2f}")


if __name__ == "__main__":
    # Example usage (Windows paths)
    input_csv_dir = r"E:\RFID-DATA\action recognition\14\17"
    output_npy_dir = r"E:\rfid_processed_npy"

    # IMPORTANT: set these to match your dataset naming protocol
    subject_id = "10"
    action_id = "17"

    batch_convert_rfid_csv(
        csv_dir=input_csv_dir,
        output_dir=output_npy_dir,
        subject_id=subject_id,
        action_id=action_id,
        repeats_per_action=20,
        target_frame=148,
        expected_max_tags=24,
        denoise=True,
        sigma_factor=1.5,
        save_plot=False,       # set True if you want diagnostic plots
        plot_dir=None
    )
