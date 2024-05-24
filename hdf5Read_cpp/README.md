# __C++ code to read HDF5 files__

The first thing to do is install the HDF5 library for C++. For simplicity you can use the following script. It will take some time.
```
./install_hdf5
```

In this folder you can find two codes. The two codes read a dataset from an HDF5 file, but they use different approaches to do so. Here's a breakdown of the key differences:
* `hdf5read_standard.cpp`: This code reads the entire dataset in a single operation, loading all the data into memory at once.
* `hdf5read_chunked.cpp`: This code reads the dataset in chunks, reducing memory use by loading only part of the dataset at a time.

Once you're done you can compile the C++ codes, if you haven't already done:
```
g++ -o hdf5read_chunked hdf5read_chunked.cpp -lhdf5_cpp -lhdf5
g++ -o hdf5read_standard hdf5read_standard.cpp -lhdf5_cpp -lhdf5
```

Finally you can run your test with the standard mode
```
./hdf5read_standard <file_name> <dataset_name>
```
or the chunked mode
```
./hdf5read_chunked <file_name> <dataset_name>
```

## __Test performance__
In ordedr to test the performance of the these two approaches you can run the following experiments:

### Memory
1. Standard Read approach
    ```
    $ ./Test/peakmem.sh hdf5read_standard ../Data/dl1new.h5
    Peak virtual memory usage: 212380 KB
    Peak Resident Set Size usage: 171640 KB
    Peak Shared Memory usage: 6608 KB
    ```
    ```
    $ ./Test/peakmem.sh hdf5read_standard ../Data/dl1chunked.h5
    Peak virtual memory usage: 165408 KB
    Peak Resident Set Size usage: 141404 KB
    Peak Shared Memory usage: 6584 KB
    ```
2. Chunked Read approach
    ```
    $ ./Test/peakmem.sh hdf5read_chunked ../Data/dl1new.h5
    Peak virtual memory usage: 213176 KB
    Peak Resident Set Size usage: 164580 KB
    Peak Shared Memory usage: 6424 KB
    ```
    ```
    $ ./Test/peakmem.sh hdf5read_chunked ../Data/dl1chunked.h5
    Peak virtual memory usage: 30644 KB
    Peak Resident Set Size usage: 6584 KB
    Peak Shared Memory usage: 6016 KB
    ```