import streamlit as st
import os
import numpy as np
from tifffile import imwrite
import tempfile
import zipfile

# ==============================
# CONFIG
# ==============================
BANDS = 141

# ==============================
# HSD READERS
# ==============================

def try_read(buffer, X, Y, Z, dtype):
    """Try reading HSD with given shape"""
    bytes_per_pixel = np.dtype(dtype).itemsize
    RAW_len = X * Y * Z * bytes_per_pixel

    if len(buffer) < RAW_len:
        raise ValueError("Too small")

    header_size = len(buffer) - RAW_len
    header = buffer[:header_size]

    dat = np.frombuffer(buffer[header_size:], dtype=dtype)

    if dat.size != X * Y * Z:
        raise ValueError("Shape mismatch")

    cube = np.reshape(dat, (Y, Z, X))
    return cube.transpose(0, 2, 1), header, Y, X


def auto_read_HSD(file_path):
    """Auto-detect HSD format"""
    with open(file_path, "rb") as f:
        buffer = f.read()

    # Known camera resolutions
    shapes = [
        (1920, 1080, BANDS),
        (1280, 1024, BANDS),
        (640, 480, BANDS),
    ]

    dtypes = [np.uint16, np.uint8, np.float32]

    for X, Y, Z in shapes:
        for dtype in dtypes:
            try:
                cube, header, Y, X = try_read(buffer, X, Y, Z, dtype)
                print(f"Detected format: {X}x{Y}x{Z}, {dtype}")
                return cube, header, Y, X
            except Exception:
                continue

    raise ValueError("Unknown HSD format")


# ==============================
# STREAMLIT UI
# ==============================

st.title("HSD → 16-bit RGB Converter")

st.sidebar.header("Band Selection")
BAND_R = st.sidebar.number_input("Red Band", value=60)
BAND_G = st.sidebar.number_input("Green Band", value=40)
BAND_B = st.sidebar.number_input("Blue Band", value=20)

uploaded_zip = st.file_uploader("Upload ZIP with HSD files", type="zip")

if uploaded_zip:

    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    output_dir = os.path.join(temp_dir, "rgb_output")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    skipped = 0

    for root, _, files in os.walk(temp_dir):
        for file in files:

            # Skip Mac hidden files
            if file.startswith("._") or file.startswith("."):
                continue

            if file.lower().endswith(".hsd"):

                path = os.path.join(root, file)

                # Skip tiny files
                if os.path.getsize(path) < 1000000:
                    skipped += 1
                    continue

                try:
                    cube, header, Y, X = auto_read_HSD(path)

                    rgb = np.stack([
                        cube[:, :, int(BAND_R)],
                        cube[:, :, int(BAND_G)],
                        cube[:, :, int(BAND_B)]
                    ], axis=-1)

                    save_path = os.path.join(
                        output_dir,
                        file.replace(".hsd", ".tiff")
                    )

                    rgb16 = rgb.astype(np.uint16)
                    imwrite(save_path, rgb16)

                    st.write(f"Preview: {file}")
                    st.image(rgb16, clamp=True)

                    count += 1

                except Exception as e:
                    st.error(f"{file} failed: {e}")

    st.success(f"Converted {count} files. Skipped {skipped} small/hidden files.")
