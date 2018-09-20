import os

from program_variables import file_loc_vars

drive_path = file_loc_vars.basic_dir
ready_path = file_loc_vars.ready_dir

generators = ['Herwig Angular', 'Herwig Dipole', 'Sherpa',
              'Pythia Standard', 'Pythia Vincia']


def get_ready_path(gen):
    # type: (str) -> str
    """
    :param gen: A generator which data path is to be returned.
    :return: Path to prepared data of a given generator.
    """
    gen_path = ready_path + gen.replace(' ', '/') + '/data.h5'
    if not os.path.exists(gen_path):
        raise IOError("Generator " + gen + " not found, at path: " + gen_path)
    return gen_path
