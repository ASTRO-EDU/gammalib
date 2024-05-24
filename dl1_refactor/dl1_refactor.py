import xml.etree.ElementTree as ET
import h5py
import numpy as np

class DL1_producer:
        def __init__(self) -> None:
            pass

        def __refactor(sel, transform_code, x):
            # Creiamo un contesto per eseguire il codice della funzione
            local_context = {}

            # Eseguiamo il codice della funzione nel contesto locale
            exec(transform_code, globals(), local_context)

            # Recuperiamo la funzione definita nel contesto locale
            transform_function = local_context['transform']

            # Ora possiamo usare la funzione
            result = transform_function(x)
            return result

        def invert_DL1(self, xml_file, inputDL1_file, outputDL1_file, use_chunks, chunks=10):
            tree = ET.parse(xml_file)
            model_root = tree.getroot()
            with h5py.File(inputDL1_file, 'r') as input_hdf:
                with h5py.File(outputDL1_file, 'w') as output_hdf:
                    # There should be just one Group "/waveforms"
                    model_group = model_root.find('Group')
                    model_groupname  = model_group.get('name')
                    # Create the same group in new DL1
                    output_group = output_hdf.create_group(model_groupname)
                    num_waveforms = len(input_hdf[model_groupname])
                    # print(f'num_waveforms = {num_waveforms}')
                    # First iteration to create the new CArray in the output file
                    model_carraylist = model_group.findall('CArray')
                    # print(f'model_carraylist = {model_carraylist}')
                    for model_carray in model_carraylist:
                        model_carrayname      = model_carray.get('name')
                        model_carraydtype     = model_carray.get('dtype')
                        model_carraycomplevel = int(model_carray.get('complevel'))
                        model_carraycomplib   = model_carray.get('complib')
                        model_carrayattrs = model_carray.findall('Attributes/Attribute')
                        num_attributes = len(model_carrayattrs)
                        column_names = [attr.get('name') for attr in model_carrayattrs]
                        # print(column_names)
                        if num_attributes == 0:
                            # Data case
                            newarray = np.zeros((num_waveforms, 16384))
                        else:
                            # Attributes case
                            newarray = np.zeros((num_waveforms, num_attributes))
                        # Iterate over all the waveforms
                        for wfname in input_hdf[model_groupname]:
                            # Get the wf_idx `wfname` has the following model wf_000005
                            wf_idx = int(wfname.split('_')[1])
                            if num_attributes == 0:
                                data = input_hdf[model_groupname][wfname]
                                newarray[wf_idx, :len(data)] = data[:, 0]
                                if use_chunks:
                                    chunkshape = (chunks, 16384)
                                else:
                                    chunkshape = (len(input_hdf[model_groupname]), 16384)
                            else:
                                # If use_chunks is set to True
                                if use_chunks:
                                    chunkshape = (chunks, len(model_carrayattrs))
                                else:
                                    chunkshape = (len(input_hdf[model_groupname]), len(model_carrayattrs))
                                for j, model_attr in enumerate(model_carrayattrs):
                                    model_attrname = model_attr.get('name')
                                    transform_code = model_attr.get('transformation')
                                    try:
                                        attr_value = input_hdf[model_groupname][wfname].attrs[model_attrname]
                                        if not transform_code is None:
                                            attr_value = self.__refactor(transform_code, attr_value)
                                        newarray[wf_idx, j] = attr_value
                                    except Exception:
                                        continue
                        
                        # Determine the numpy dtype
                        if model_carraydtype == 'int16':
                            np_dtype = np.int16
                        elif model_carraydtype == 'int64':
                            np_dtype = np.int64
                        elif model_carraydtype == 'float64':
                            np_dtype = np.float64
                        else:
                            raise ValueError(f"Unsupported dtype: {model_carraydtype}")
                        
                        dataset = output_group.create_dataset(model_carrayname, data=newarray,
                                                        dtype=np_dtype, compression=model_carraycomplib,
                                                        compression_opts=model_carraycomplevel,
                                                        chunks=chunkshape)
                        # Store column names as a single attribute
                        dataset.attrs['column_names'] = ','.join(column_names)