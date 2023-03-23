import ase.io
from ase import Atoms
from AtomicAI.io.read_cq import read_cq

"""
    Reading the input files for structure conversion and plotting RDF, DOS, etc.,
"""
def read(input_file: str = None, 
        structure_conversion: bool = False,
        file_format: str = None,
        plot_dos: bool = False, 
        plot_rdf: bool = False,
        ):
    if input_file == None:
        print(f'No input file found. Check input format!!!')
        exit()
    file_type = input_file.split('.')[-1]
    if structure_conversion:
        acceptable_filetypes = list(ase.io.formats.ioformats.keys())  # Many input out formats are available  
        if file_type in acceptable_filetypes:
            ase_data = ase.io.read(filename, format=file_type)
        elif file_format == 'cq':
            ase_data = read_cq(input_file)


        else:
            print(f'Given file format {file_type} is not supported by ase. \n')
            print(f'Acceptable formats are{list(ase.io.formats.ioformats.keys())} \n and cq')
            exit()

        return ase_data
