#!/usr/bin/env python
#################################################
""" helpful_python.py
# # Collection of useful Python functions.
#   Function categories:
#     Dictionary:
#       - Building, sorting, extracting, processing, printing.
#     Number Manipulation:
#      - Hex, binary.
#     Relative Processing:
#      - Sorting, filtering, extraction.
#     Sampling:
#      - Chunking, rebin.
#     Validation:
#      - Check func input, Python ver.
#     Data Handling:
#      - Save, load objects.
#     String Manipulation:
#      - Tokenization, replace multiple.
#     Data Transformation:
#      - Norm, standard, log, binary.
#     Perofrmance Measurment:
#      - Perofrmance metrics, confusion table.
#     Machine Learning:
#      - Convolution, pool output size.
#     Misc:
#      - Create dir, remove duplicates, operators.
"""
#################################################
# ###  Author: Samyuel Danyo
# ###  Date: 24/03/2020
# ###  Last Edit: 28/06/2020
#################################################
# coding: utf-8
# ## imports
# Python Standard Library
import math

# Third Party Imports

# Local Application/Library Specific Imports.

#################################################
# Number Manipulation
#################################################
def hex_to_hex(num, short=False):
    """ Return hex number from prefixed(0x) hex number string."""
    num = int(num, 16)
    if short:
        return '%x' % num
    return '0x%x' % num

def to_hex(num, base=10, short=False):
    """ Return hex number from binary, decimal, hex strings/numbers."""
    try:
        if not isinstance(num, str):
            if base == 16:
                num = '{:x}'.format(num)
            elif base == 2:
                num = '{:b}'.format(num)

        if short:
            return '%x' % int('{}'.format(num), base)
        return '0x%x' % int('{}'.format(num), base)
    except Exception:
        return hex_to_hex(num, short)

def least_sig_bits_hex(src, bits, out_t=int, arr=True):
    """ Get the bits least significant bits of a/list of hex number/s.
        Args:
            src (String/List of String): Source hex numbers.
            bits (Integer): Number of least-sig bits.
            out_t (int or hex): Output type. Integer and Hex supported.
        Returns:
            out (out_t/List(out_t)): The least-sig bits for each el in src."""
    # List
    if arr:
        try:
            out = []
            for elem in src:
                # Transform to int
                elem = int(elem, 16)
                # Transform to binary & get least sig bits.
                elem = bin(elem)
                # Get least sig bits (check bits is not more than number).
                elem = elem[-min(bits, len(elem)-2):]
                # Append the number
                if out_t is int or out_t == 'int':
                    out.append(int(elem, 2))
                else:
                    out.append(to_hex(elem, base=2, short=True))
            return out
        except:
            pass
    # Single number
    # Transform to int
    src = int(src, 16)
    # Transform to binary
    src = bin(src)
    # Get least sig bits (check bits is not more than number).
    src = src[-min(bits, len(src)-2):]
    # Return the number
    if out_t is int or out_t == 'int':
        return int(src, 2)
    return to_hex(src, base=2, short=True)

def append_bits(src, append_src, out_t=int):
    """ Append bits to src integer/binary numbers."""
    # List
    try:
        if len(src) != len(append_src):
            print("ERROR:: src and append_src need to have the same lenght!!!")
            return False
        out = []
        for elem, bits in zip(src, append_src):
            if out_t is int or out_t == 'int':
                out.append(int(bin(elem)+'{}'.format(bits), 2))
            else:
                out.append(elem+str(bits))
        return out
    # Single number
    except Exception:
        if out_t is int or out_t == 'int':
            return int(bin(elem)+'{}'.format(bits), 2)
        return elem+str(bits)

def convert_byte_size(size_bytes):
    """Convert {size_bytes} bytes to string with degree literal.
       Support all base systems: {hex-B16-0xn,
                                  interger-B10-n,
                                  octal-B8-0on,
                                  binary-B2-0bn}."""
    if size_bytes == 0:
        return "0B"
    degree_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    degree = int(math.floor(math.log(size_bytes, 1024)))
    degree_order = math.pow(1024, degree)
    bytes_in_degree = round(size_bytes / degree_order, 2)
    if bytes_in_degree % 1 == 0:
        bytes_in_degree = int(bytes_in_degree)
    return "%s%s" % (bytes_in_degree, degree_name[degree])
