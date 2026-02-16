import streamlit as st
import os
import numpy as np
from tifffile import imwrite
import tempfile
import zipfile

import numpy as np
#This code provides functions to read and write hyper-spectral data (HSD) from/to binary files, including handling header information that may be present in the files.
 
 #What is a header?
# a "header" refers to the metadata or descriptive information that is stored at the beginning of a binary file before the actual data. This header typically contains information necessary to interpret the data correctly, such as:
# File format version
# Data dimensions (e.g., width, height, number of bands)
# Data type (e.g., uint16, uint8)


# Acquisition parameters (e.g., timestamp, sensor settings)
# Any other relevant metadata

#metadata
#This metadata helps any software or code processing the file to correctly interpret and manipulate the data that follows the header.

#.hsd file format
#The .hsd file format typically includes:

#Header: Metadata about the hyper-spectral data, such as dimensions, data type, wavelength information, and other relevant parameters.
#Data: The actual hyper-spectral data, which is usually stored as a multi-dimensional array. This array could have dimensions such as (bands, rows, columns) or (rows, columns, bands), depending on the specific application and storage requirements.

#The need for reading different read funstions for different file sizes()
#arises because different types of hyper-spectral cameras or data formats may produce files with varying structures, metadata, and data encoding schemes. The file size can be an indicator of which format or version of the data is being dealt with.

#buffer
#A buffer in the context of computing and data processing is a temporary storage area used to hold data while it is being transferred from one place to another. 

#Tuple
#A tuple is an ordered collection of elements in Python that is immutable, meaning once a tuple is created, its contents cannot be changed. Tuples can hold elements of different data types and are defined using parentheses ().

#Summary
#save_HSD_with_header: Saves HSD data along with its header into a binary file.
#read_HSD_from_file: Reads HSD data from a file, determining the appropriate reader function based on file size.
#read_HSC180X_CL, read_HSC180X, read_HSC170X_old, read_HSC170X_new: These functions handle specific data formats, extracting the header and data, and reshaping the data into the correct dimensions.

#advantages of using binary file
#Efficiency:Speed of Access, Memory Usage
#Storage:Compact Representation
#Precision:Exact Representation
#Compatibility: Interoperability, Standardization
#Performance: Reduced Overhead, Parallel Processing

# Created on 2024-03-07
# Implemented separation and output of binary information from the header
# Implemented save_HSD_with_header function to restore the original binary data by combining header information and a numpy array
# Supports reading HSC1804CL (Zero Type)

def save_HSD_with_header(file_path: str, data: np.ndarray, header: bytes) -> None:
    """
    Save hyper-spectral data (HSD) with its header into a binary file.
    
    Args:
        file_path (str): The path to the file where the data should be saved.
        data (np.ndarray): The HSD data to be saved.
        header (bytes): The header information to be prepended to the data.
    """
    # Transpose the data to its original shape before saving
    data = data.transpose(0, 2, 1)#	Therefore, HSData.transpose(0, 2, 1) swaps the second and third dimensions of the HSData array. If the original shape of HSData was (A, B, C), the transposed array will have a shape of (A, C, B), effectively swapping the dimensions corresponding to axes 1 and 2.
    
    # Convert the numpy array data to bytes
    data_bytes = data.tobytes()
    
    # Write the header and data to the file
    with open(file_path, 'wb') as file:
        file.write(header + data_bytes)


def read_HSD_from_file(file_path: str, band: int = 141) -> tuple:
    """
    Read hyper-spectral data (HSD) from a file and return it along with the header information.
    
    Args:
        file_path (str): The path to the .dat or .hsd file to be read.
        band (int): The number of bands in the HSD data. Default is 141.
        
    Returns:
        tuple: A tuple containing the HSD data as a numpy array, the header information, and the dimensions Y and X.
    """
    file_extension = file_path.split('.')[-1]
    #The function determines the file extension and reads the file into a buffer.
    with open(file_path, 'rb') as file:
        buffer = file.read()
    #Depending on the file size, it calls the appropriate function to process the data (e.g., read_HSC180X, read_HSC170X_old, etc.).
    if file_extension in ['hsd', 'dat']:
        file_size = len(buffer)
        if file_size == 370623040:
            HSData, header, Y, X = read_HSC180X(buffer)
        elif file_size == 87630400:
            HSData, header, Y, X = read_HSC170X_old(buffer)
        elif file_size == 44315200:
            HSData, header, Y, X = read_HSC170X_new(buffer)
        elif file_size == 585755200:
            HSData, header, Y, X = read_HSC180X_CL(buffer)
        else:
            raise ValueError(f"Unsupported file size: {file_size}.")
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}.")
    
    return HSData.transpose(0, 2, 1), header, Y, X


def read_HSC180X_CL(buffer: bytes) -> tuple:
    """
    Read HSC180X_CL hyper-spectral data from a buffer.
    
    Args:
        buffer (bytes): The binary data buffer.
        
    Returns:
        tuple: A tuple containing the HSD data as a numpy array, the header information, and the dimensions Y and X.
    """
    X, Y, Z = 1920, 1080, 141#Defines the dimensions X, Y, Z.
    RAW_len = X * Y * Z * 2#Calculates the raw data length (RAW_len).
    header = buffer[:len(buffer) - RAW_len]#Extracts the header and prints its size.
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint16)#Converts the data part of the buffer from bytes to a numpy array.
    HSData = np.reshape(dat, (Y, Z, X))#Reshapes the data to the correct dimensions.
    return HSData, header, Y, X


def read_HSC180X(buffer: bytes) -> tuple:
    """
    Read HSC180X hyper-spectral data from a buffer.
    
    Args:
        buffer (bytes): The binary data buffer.
        
    Returns:
        tuple: A tuple containing the HSD data as a numpy array, the header information, and the dimensions Y and X.
    """
    X, Y, Z = 1280, 1024, 141
    RAW_len = X * Y * Z * 2#16-Bit Integers: Each element of the hyper-spectral data array is assumed to be represented by a 16-bit integer (also known as uint16 in numpy), which occupies 2 bytes of memory. This assumption is made based on the data format or specifications of the HSC180X hyper-spectral data.
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint16)
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, header, Y, X


def read_HSC170X_old(buffer: bytes) -> tuple:
    """
    Read old HSC170X hyper-spectral data from a buffer.
    
    Args:
        buffer (bytes): The binary data buffer.
        
    Returns:
        tuple: A tuple containing the HSD data as a numpy array, the header information, and the dimensions Y and X.
    """
    X, Y, Z = 640, 480, 141
    RAW_len = X * Y * Z * 2
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint16)
    dat = dat.astype(np.uint8)#Converts the data to uint8 type.
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, header, Y, X


def read_HSC170X_new(buffer: bytes) -> tuple:
    """
    Read new HSC170X hyper-spectral data from a buffer.
    
    Args:
        buffer (bytes): The binary data buffer.
        
    Returns:
        tuple: A tuple containing the HSD data as a numpy array, the header information, and the dimensions Y and X.
    """
    X, Y, Z = 640, 480, 141
    RAW_len = X * Y * Z
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint8)
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, header, Y, X

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

    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".hsd"):
                try:
                    cube, header, Y, X = read_HSD_from_file(os.path.join(root, file))

                    rgb = np.stack([
                        cube[:, :, BAND_R],
                        cube[:, :, BAND_G],
                        cube[:, :, BAND_B]
                    ], axis=-1)

                    save_path = os.path.join(output_dir, file.replace(".hsd", ".tiff"))
                    imwrite(save_path, rgb)
                    count += 1

                except Exception as e:
                    st.error(f"{file} failed: {e}")

    st.success(f"Converted {count} files!")

