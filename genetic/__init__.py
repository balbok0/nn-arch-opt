import sys, os
sys.path.append(os.path.realpath(__file__)[:-11])

from program_variables import program_params
from mutator import Mutator
from network import Network
from helpers.helpers_data import prepare_data