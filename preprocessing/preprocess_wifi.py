import os
import numpy as np
from tqdm import tqdm
import pywt   # Wavelet library: pip install PyWavelets


# --------- 1. CSI parsing functions (validated logic) ---------
def parse_csi(payload, Ntx, Nrx):
    """
    Parse raw payload into a complex CSI matrix.
    Assumption:
      - 30 subcarriers
      - Each subcarrier contains 1 byte for real and 1 byte for imaginary part

    Output shape: (1, Nrx, 30)
    """
    index = 0
    csi = np.zeros((1, Nrx, 30), dtype=complex)

    for subcarrier_index in range(30):
        for rx in range(Nrx):
            if index + 1 >= len(payload):
                csi[0, rx, subcarrier_index] = 0
                continue

            real = payload[index]
            imag = payload[index + 1]

            if real > 127:
                real -= 256
            if imag > 127:
                imag -= 256

            csi[0, rx, subcarrier_index] = complex(real, imag)
            index += 2

    return csi


def read_bf_file(filename):
    """
    Read a single .dat file and parse all CSI frames.

    Returns:
      A list of dictionaries:
        [{'csi': (1, Nrx, 30)}, ...]
    """
    with open(filename, "rb") as f:
        bfee_list = []
        field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
        while field_len != 0:
            bfee_list.append(f.read(field_len))
            field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)

    entries = []
    triangle = [0, 1, 3]  # Antenna permutation validation

    for array in bfee_list:
        if array[0] != 187:   # 0xbb
            continue

        Nrx = array[9]
        Ntx = array[10]
        antenna_sel = array[16]
        payload = array[21:]

        perm = [
            (antenna_sel & 0x3),
            ((antenna_sel >> 2) & 0x3),
            ((antenna_sel >> 4) & 0x3)
        ]

        csi = parse_csi(payload, Ntx, Nrx)

        # Simple antenna reordering
        if sum(perm) == triangle[Nrx - 1]:
            csi[:, perm, :] = csi[:, [0, 1, 2], :]

        entries.append({"csi": csi})

    return entries


def extract_csi_matrix(csi_entry):
    """
    Extract CSI amplitude matrix from one entry.

    Output shape: (3, 30)
    """
    return np.abs(np.array(csi_entry['csi']).squeeze())


# --------- 2. Wavelet denoising functions (along temporal axis) ---------
def wavelet_denoise_1d(x, wavelet='db4', level=None, mode='soft'):
    """
    Perform wavelet threshold denoising on a 1D signal.

    Steps:
      1) Wavelet decomposition
      2) Thresholding on detail coefficients
      3) Wavelet reconstruction

    Args:
      x: 1D numpy array
    """
    coeffs = pywt.wavedec(x, wavelet, mode='per', level=level)

    # Estimate noise level using the highest-frequency detail coefficients
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # Universal threshold
    uthr = sigma * np.sqrt(2 * np.log(len(x)))

    # Apply thresholding only to detail coefficients
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_c = pywt.threshold(c, uthr, mode=mode)
        new_coeffs.append(new_c)

    x_rec = pywt.waverec(new_coeffs, wavelet, mode='per')

    # Crop to original length
    return x_rec[:len(x)]


def wavelet_denoise_csi(csi_array, wavelet='db4', level=None, mode='soft'):
    """
    Apply wavelet denoising to CSI amplitude data along the temporal axis.

    Input shape:  (T, 3, 30)
    Output shape: (T, 3, 30)
    """
    T, Nrx, Nsc = csi_array.shape
    denoised = np.zeros_like(csi_array)

    for rx in range(Nrx):
        for sc in range(Nsc):
            x = csi_array[:, rx, sc]
            denoised[:, rx, sc] = wavelet_denoise_1d(
                x,
                wavelet=wavelet,
                level=level,
                mode=mode
            )

    return denoised


# --------- 3. Single .dat file parsing + optional denoising ---------
def parse_dat_file(file_path,
                   expected_frames=1200,
                   denoise=True,
                   wavelet='db4',
                   level=None,
                   mode='soft'):
    """
    Process a single .dat file:
      1) Parse CSI frames with shape (3, 30)
      2) Stack into (T, 3, 30) and truncate to expected_frames
      3) Optionally apply wavelet denoising along the temporal axis

    Returns:
      csi_array with shape (T, 3, 30)
    """
    entries = read_bf_file(file_path)
    csi_list = []

    for entry in entries:
        if 'csi' not in entry:
            continue
        csi = extract_csi_matrix(entry)
        if csi.shape == (3, 30):
            csi_list.append(csi)

    csi_array = np.array(csi_list)
    csi_array = csi_array[:expected_frames]

    if csi_array.size == 0:
        return csi_array

    if denoise:
        csi_array = wavelet_denoise_csi(
            csi_array,
            wavelet=wavelet,
            level=level,
            mode=mode
        )

    return csi_array


# --------- 4. Three-device fusion: denoise first, then concatenate ---------
def merge_csi_three_devices(dat1, dat2, dat3,
                            expected_frames=1200,
                            denoise=True,
                            wavelet='db4',
                            level=None,
                            mode='soft'):
    """
    Process CSI data from three receivers:
      - Parse each device independently
      - Optionally apply wavelet denoising
      - Align frame numbers
      - Concatenate along the Rx dimension

    Output:
      merged array with shape (270, T)
    """
    csi1 = parse_dat_file(dat1, expected_frames, denoise, wavelet, level, mode)
    csi2 = parse_dat_file(dat2, expected_frames, denoise, wavelet, level, mode)
    csi3 = parse_dat_file(dat3, expected_frames, denoise, wavelet, level, mode)

    assert csi1.shape == csi2.shape == csi3.shape, \
        "Frame count or shape mismatch among the three devices"

    merged = []
    for i in range(csi1.shape[0]):
        merged_frame = np.concatenate([csi1[i], csi2[i], csi3[i]], axis=0)
        merged.append(merged_frame.reshape(-1))

    merged = np.array(merged).T  # (270, frames)
    return merged


# --------- 5. Batch processing for a full action sequence ---------
def batch_process(dev1_dir, dev2_dir, dev3_dir,
                  output_dir, action_id,
                  expected_frames=1200,
                  denoise=True,
                  wavelet='db4',
                  level=None,
                  mode='soft'):

    os.makedirs(output_dir, exist_ok=True)

    files1 = sorted([f for f in os.listdir(dev1_dir) if f.endswith(".dat")])
    files2 = sorted([f for f in os.listdir(dev2_dir) if f.endswith(".dat")])
    files3 = sorted([f for f in os.listdir(dev3_dir) if f.endswith(".dat")])

    assert len(files1) == len(files2) == len(files3), \
        "The number of files differs across devices"

    for idx, (f1, f2, f3) in enumerate(
            tqdm(zip(files1, files2, files3),
                 total=len(files1),
                 desc="Processing")):

        path1 = os.path.join(dev1_dir, f1)
        path2 = os.path.join(dev2_dir, f2)
        path3 = os.path.join(dev3_dir, f3)

        merged = merge_csi_three_devices(
            path1, path2, path3,
            expected_frames=expected_frames,
            denoise=denoise,
            wavelet=wavelet,
            level=level,
            mode=mode
        )

        # File naming: <subject>_<action>_<index>.npy, e.g., 23_18_01.npy
        filename = f"23_{action_id}_{idx + 1:02d}.npy"
        out_path = os.path.join(output_dir, filename)
        np.save(out_path, merged)

    print(f"[Done] Processed {len(files1)} action samples. Output saved to {output_dir}")


# --------- 6. Example entry point ---------
if __name__ == "__main__":
    subject_id = "23"
    action_id = "18"
    subfolder_id = "18"

    dev1_path = f"E://{subject_id}//3//xingwei//{subfolder_id}"
    dev2_path = f"E://{subject_id}//4//xingwei//{subfolder_id}"
    dev3_path = f"E://{subject_id}//5//xingwei//{subfolder_id}"

    output_path = "E://wavelet_wifi_processed"

    batch_process(
        dev1_dir=dev1_path,
        dev2_dir=dev2_path,
        dev3_dir=dev3_path,
        output_dir=output_path,
        action_id=action_id,
        expected_frames=1200,
        denoise=True,
        wavelet='db4',
        level=None,
        mode='soft'
    )
