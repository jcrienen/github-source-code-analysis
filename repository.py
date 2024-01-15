import os
from enum import Enum
import configuration
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer


class AnalysisType(Enum):
    ALL = "all"
    CLASS_NAMES = configuration.config["files.class-names-filename"].data
    METHOD_NAMES = configuration.config["files.method-names-filename"].data
    GLOBAL_VARIABLES = configuration.config["files.global-variables-filename"].data
    LOCAL_VARIABLES = configuration.config["files.local-variables-filename"].data
    PARAMETERS = configuration.config["files.parameters-filename"].data


class Repository:
    def __load_all(self, folder):
        data = {}
        # self.target = folder.name.split(" - ")[2]

        for file in os.scandir(folder):
            with open(file, "r") as f:

                filename = os.path.splitext(file.name)[0]
                data[filename] = []
                for line in f:
                    line = line.strip()
                    data[filename].append(line)

        return data

    def __init__(self, folder):
        raw = self.__load_all(folder)

        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english')).union(configuration.config["stopwords"].data.split(","))

        self.data = {}

        for analysis_type in raw.keys():
            data = [i.lower() for i in raw[analysis_type]]
            stemmed_data = [stemmer.stem(i) for i in data]

            self.data[analysis_type] = " ".join([w for w in stemmed_data if len(w) > 3])


class Repositories:
    def __init__(self):
        self.data = {}

        for folder in os.scandir(configuration.config["output-folder"].data):
            if os.path.isdir(folder):
                self.data[folder.name] = Repository(folder)
