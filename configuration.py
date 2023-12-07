from jproperties import Properties

config = Properties()


def load_properties(file):
    global config

    with open(file, 'rb') as config_file:
        config.load(config_file)
