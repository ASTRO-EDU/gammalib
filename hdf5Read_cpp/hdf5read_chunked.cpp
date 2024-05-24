#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
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
    // Questa sleep serve ai fini della test per il calcolo dell'uso della memoria nella versione chunked
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

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

        // Ottieni le dimensioni del chunk
        DSetCreatPropList plist = dataset.getCreatePlist();
        hsize_t chunk_dims[rank];
        plist.getChunk(rank, chunk_dims);

        // Buffer per leggere i dati di un chunk
        std::vector<double> data(chunk_dims[0] * chunk_dims[1]);

        // Calcola il numero di chunk in ogni dimensione
        hsize_t num_chunks[rank];
        for (int i = 0; i < rank; ++i) {
            num_chunks[i] = (dims[i] + chunk_dims[i] - 1) / chunk_dims[i];
        }

        // Leggi i dati chunk per chunk
        for (hsize_t i = 0; i < num_chunks[0]; ++i) {
            for (hsize_t j = 0; j < num_chunks[1]; ++j) {
                // Calcola l'offset del chunk
                hsize_t offset[2] = {i * chunk_dims[0], j * chunk_dims[1]};
                
                // Calcola le dimensioni effettive del chunk (possono essere minori dei chunk_dims ai bordi)
                hsize_t actual_chunk_dims[2] = {
                    std::min(chunk_dims[0], dims[0] - offset[0]),
                    std::min(chunk_dims[1], dims[1] - offset[1])
                };

                // Seleziona l'ipotesi di lettura
                dataspace.selectHyperslab(H5S_SELECT_SET, actual_chunk_dims, offset);

                // Crea uno spazio in memoria per il chunk
                DataSpace memspace(2, actual_chunk_dims);

                // Leggi il chunk
                dataset.read(data.data(), PredType::NATIVE_DOUBLE, memspace, dataspace);

                // Stampa i dati
                // for (hsize_t x = 0; x < actual_chunk_dims[0]; ++x) {
                //     for (hsize_t y = 0; y < actual_chunk_dims[1]; ++y) {
                //         std::cout << data[x * chunk_dims[1] + y] << " ";
                //     }
                //     std::cout << std::endl;
                // }
            }
        }
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
