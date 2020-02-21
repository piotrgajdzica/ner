import os


def all_files(directory):
    if os.path.isfile(directory):
        yield directory
    else:
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            if os.path.isfile(path):
                yield path
            elif os.path.isdir(path):
                for file in all_files(path):
                    yield file
