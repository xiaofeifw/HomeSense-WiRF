import os
import shutil
from tqdm import tqdm


def split_train_test_new(
    root_ns: str = "./dataset/Raw_dataset/",
    dst_wr: str = "./dataset/HS-WIRF_dataset/",
    split: int = 14
):
    """
    Split dataset into train/test by repetition index (actidx).
    For each sample, if actidx <= split -> train, else -> test.

    Expected structure:
      root_ns/
        RFID/*.npy
        WiFi/*.npy

    Output structure:
      dst_wr/
        train_data/RFID/
        train_data/WiFi/
        test_data/RFID/
        test_data/WiFi/
    """

    # --- Source dirs (only RFID + WiFi) ---
    src_rfid = os.path.join(root_ns, "RFID")
    src_wifi = os.path.join(root_ns, "WiFi")

    if not os.path.isdir(src_rfid):
        raise FileNotFoundError(f"RFID folder not found: {src_rfid}")
    if not os.path.isdir(src_wifi):
        raise FileNotFoundError(f"WiFi folder not found: {src_wifi}")

    # --- Destination dirs (only RFID + WiFi) ---
    dst_train_rfid = os.path.join(dst_wr, "train_data", "RFID")
    dst_train_wifi = os.path.join(dst_wr, "train_data", "WiFi")
    dst_test_rfid = os.path.join(dst_wr, "test_data", "RFID")
    dst_test_wifi = os.path.join(dst_wr, "test_data", "WiFi")

    for d in [dst_train_rfid, dst_train_wifi, dst_test_rfid, dst_test_wifi]:
        os.makedirs(d, exist_ok=True)

    # --- Iterate over RFID files as the index ---
    rfid_files = [f for f in os.listdir(src_rfid) if f.endswith(".npy")]

    for file in tqdm(rfid_files, desc="Splitting"):
        filename = os.path.splitext(file)[0]  # remove .npy

        # filename pattern assumed: <personidx>_<...>_<actidx>_...
        parts = filename.split("_")
        if len(parts) < 3:
            # skip unexpected filenames
            continue

        try:
            actidx = int(parts[2])
        except ValueError:
            # skip if actidx not parseable
            continue

        src_rfid_path = os.path.join(src_rfid, filename + ".npy")
        src_wifi_path = os.path.join(src_wifi, filename + ".npy")

        # Ensure paired WiFi exists
        if not os.path.isfile(src_wifi_path):
            # If you prefer hard-fail, replace with: raise FileNotFoundError(...)
            print(f"[WARN] Missing WiFi pair for: {filename}.npy -> skipped")
            continue

        if actidx <= split:
            shutil.copy(src_rfid_path, os.path.join(dst_train_rfid, filename + ".npy"))
            shutil.copy(src_wifi_path, os.path.join(dst_train_wifi, filename + ".npy"))
        else:
            shutil.copy(src_rfid_path, os.path.join(dst_test_rfid, filename + ".npy"))
            shutil.copy(src_wifi_path, os.path.join(dst_test_wifi, filename + ".npy"))


if __name__ == "__main__":
    split_train_test_new(root_ns="./dataset/Raw_dataset/", dst_wr="./dataset/HS-WIRF_dataset/", split=14)
