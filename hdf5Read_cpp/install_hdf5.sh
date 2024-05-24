HDF5_VERSION=1.14.1
wget --no-check-certificate https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$(echo "${HDF5_VERSION}" | sed 's/\.[0-9]*$//')/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}-2.tar.gz -q && \
    tar -xzf hdf5-${HDF5_VERSION}-2.tar.gz && \
    cd hdf5-${HDF5_VERSION}-2 && \
    ./configure --prefix=/usr/local  --enable-cxx && \
    make -j8 && \
    make install