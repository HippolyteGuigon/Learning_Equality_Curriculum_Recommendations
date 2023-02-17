import json


def load_file(path: str) -> json:
    """
    The goal of this function is to load a
    json file

    Arguments:
        -path: str: The path where the file is
        located

    Returns:
        data: json: The dictionnary that has just
        been loaded
    """

    f = open(path)
    data = json.load(f)

    return data
