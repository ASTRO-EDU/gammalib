#include <iostream>
#include <vector>
#include "H5Cpp.h"

using namespace H5;

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <file_name> <dataset_name>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string fileName = argv[1];
    const std::string datasetName = argv[2];

    try {
        // Apri il file
        H5File file(fileName, H5F_ACC_RDONLY);

        // Apri il dataset
        DataSet dataset = file.openDataSet(datasetName);

        // Ottieni lo spazio dei dati
        DataSpace dataspace = dataset.getSpace();

        // Ottieni il numero di dimensioni del dataset
        int rank = dataspace.getSimpleExtentNdims();

        // Ottieni le dimensioni del dataset
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims, NULL);

        // Crea un buffer per leggere i dati
        std::vector<double> data(dims[0] * dims[1]);

        // Leggi i dati dal dataset
        dataset.read(data.data(), PredType::NATIVE_DOUBLE);

        // Stampa i dati (assumiamo che il dataset sia 2D)
        // for (hsize_t i = 0; i < dims[0]; ++i) {
        //     for (hsize_t j = 0; j < dims[1]; ++j) {
        //         std::cout << data[i * dims[1] + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
    catch (FileIException& error) {
        error.printErrorStack();
        return -1;
    }
    catch (DataSetIException& error) {
        error.printErrorStack();
        return -1;
    }
    catch (DataSpaceIException& error) {
        error.printErrorStack();
        return -1;
    }

    return 0;
}
