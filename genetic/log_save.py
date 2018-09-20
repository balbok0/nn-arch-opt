import os
from datetime import datetime

from program_variables.file_loc_vars import log_file_name, log_file_location


def print_message(message):
    # type: (str) -> None
    """
    Prints message to a file, with a location, specified in file_loc_vars.

    :param message: Message to be printed to the file.
    """
    if not os.path.exists(log_file_location + log_file_name):
        __create_log_file()
    separator = "  -  "
    with open(log_file_location + log_file_name, mode='a') as f:
        f.write(str(datetime.now()) + separator + message + '\n')


def __create_log_file():
    """
    Creates a new txt file, with name and location specified in file_loc_vars.
    """
    if not log_file_location == '':
        if not log_file_location.endswith('/'):
            raise Exception('Invalid location of log file.')
        if not os.path.exists(log_file_location):
            os.makedirs(log_file_location)

    # Needed for creating the file
    with open(log_file_location + log_file_name, mode='w') as _:
        pass


def reset():
    """
    Creates (and overwrites, if needed) a new txt file, to which log info can be printed.
    """
    __create_log_file()
