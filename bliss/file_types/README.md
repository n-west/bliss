
# File Types

BLISS contains readers for various file types:

* hdf5 filterbank file

## HDF5 Filterbank File

This file described in (Matt's paper)

The resulting class `h5_filterbank_file` reads a file by accepting a file path. The underlying reader is mostly generic to HDF5, except that when reading the `file` attributes of `CLASS` must match `FILTERBANK` and `VERSION` must be `1.0`. Bug reports or patches to support different versions are highly encouraged and welcome. Variations on file `CLASS` are also welcome.

A `FILTERBANK` file is assumed to have two datasets named `data` and `mask`

Other dataset attributes can be read using 
