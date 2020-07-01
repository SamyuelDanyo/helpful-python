# hepful-python
Helper functions for Python magic.  
While using Python I found myself needing some short and in my opinion highly reusable functions. Hence I decided to package those and share them with you.

## Module Categories

__Dictionary:__  
```- Building, sorting, extracting, processing, printing.```  
__Number Manipulation:__  
```- Hex, binary.```  
__Relative Processing:__  
```- Sorting, filtering, extraction.```  
__Sampling:__  
```- Chunking, rebin.```  
__Validation:__  
```- Check func input, Python ver.```  
__Data Handling:__  
```- Save, load objects.```  
__String Manipulation:__  
```- Tokenization, replace multiple.```  
__Data Transformation:__  
```- Norm, standard, log, binary.```  
__Perofrmance Measurment:__  
```- Perofrmance metrics, confusion table.```  
__Machine Learning:__  
```- Convolution, pool output size.```  
__Misc:__  
```- Create dir, remove duplicates, operators.```

## Dependencies:
------------
__Find package versions in [requirements.txt](https://github.com/SamyuelDanyo/helpful-python/blob/master/docs/requirements.txt)__

    Python --version >= 3.7
    Python Standard Library: (os, sys, pickle, operator)
    Third Party: (NumPy, MatPlotLib, PyTorch, pandas, scikit-learn)

    Recommended:
        Please make sure that you have installed Anaconda Python (version >= 3.7) in your environment.
        Information on installing Anaconda:
        < https://www.anaconda.com/distribution/ >
        < https://docs.anaconda.com/anaconda/install/windows/ >
        < https://docs.anaconda.com/anaconda/install/linux/ >

    Secondary:
        If you already have Python &/or do not want to install a full distribution:
        Make sure your environment satisfies all the requirements/dependencies listed above.
        Recommended: create a virtual environment and install requirements.txt

```
#################################################
# Reccomended import: import helpful_python as hp
# Reccomended usage: hp.module.func(input)
# Information about a module: help(hp.module)
# Set output dir: hp.set_out_dir('/path/out/dir')
# Set verboseprint: hp.set_verboseprint(hp.
#                                       misc.
#                                       init_verbose_print(verbose=True,
#                                                          vfunc=print,
#                                                          nvfunc=hp.misc.log)
#################################################
```
