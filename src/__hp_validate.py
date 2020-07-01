#!/usr/bin/env python
#################################################
""" __hp_validate.py
# # Collection of useful validation, error handling Python functions.
#     - Check func input, Python ver.
"""
#################################################
# ###  Author: Samyuel Danyo
# ###  Date: 24/03/2020
# ###  Last Edit: 28/06/2020
#################################################
# coding: utf-8
# ## imports
# Python Standard Library
import sys

# Third Party Imports

# Local Application/Library Specific Imports.

#################################################
# Validation | Error Handling
#################################################
def check_python_ver(ver=3.7):
    """ Assert minimum Python version."""
    sys_ver = sys.version_info
    assert sys_ver >= (ver//1, ver%1), (
        "Update Python. Versions: Required[{}] | Current[{}]".format(ver, sys_ver))

def check(func, arg, msg):
    """ Make sure func(arg) does NOT raise an exception.
        Args:
            func (Function): Function to be cheked:
            arg  (*)       : Argument to func().
            msg  (String)  : Error message ot be printed.
        Returns:
            func(arg) if no exception occurs, None if it does."""
    try:
        arg = func(arg)
        return arg
    except Exception:
        print("ERROR:: "+msg)
        return None

def rcheck(func, arg, msg):
    """ Make sure func(arg) DOES raise an exception.
        Args:
            func (Function): Function to be cheked:
            arg  (*)       : Argument to func().
            msg  (String)  : Error message ot be printed.
        Returns:
            None if exception occurs, arg if it doesn't."""
    try:
        arg = func(arg)
        print("ERROR:: "+msg)
        return None
    except Exception:
        return arg
