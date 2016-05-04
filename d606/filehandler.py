import cPickle
import os

directory = "picklefiles"


def save_data(runs, filename):
    # Check if directory is present, or else create it
    if not os.path.isdir(directory):
        os.mkdir(directory)

    # Check if the directory contains more that 100 files
    # If so remove the least recently used files
    if check_if_cleanups_needed():
        cleanup_directory()

    # Write the data to file
    with open(directory + "/" + filename, 'wb') as output:
        cPickle.dump(runs, output, cPickle.HIGHEST_PROTOCOL)


def load_data(filename):
    with open(filename, 'rb') as input:
        runs = cPickle.load(input)
    return runs


def file_is_present(filename):
    os.chdir(directory)
    files_in_directory = filter(os.path.isfile, os.listdir(os.getcwd()))
    return True if filename in files_in_directory else False


def check_if_cleanups_needed():
    files_in_directory = filter(os.path.isfile, os.listdir(directory))
    return 0 if len(files_in_directory) < 100 else 1


def generate_filename(oacl_ranges, m, subject):
    return str(subject) + str(oacl_ranges[0][0]) + str(oacl_ranges[0][1]) + \
           str(oacl_ranges[1][0]) + str(oacl_ranges[1][1]) + str(m) + '.dump'


def cleanup_directory():
    files_in_directory = filter(os.path.isfile, os.listdir(directory))
    file_to_delete1 = files_in_directory[0].replace('evals', '')
    file_to_delete1 = file_to_delete1.replace('runs', '')
    file_to_delete2 = 'runs' + file_to_delete1
    file_to_delete1 = 'evals' + file_to_delete1
    os.remove(directory + "/" + file_to_delete1)
    os.remove(directory + "/" + file_to_delete2)
    files_in_directory.remove(file_to_delete1)
    files_in_directory.remove(file_to_delete2)
