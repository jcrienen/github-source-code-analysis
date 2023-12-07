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
    def __load_all(self, folder, split):
        self.data = []
        self.target = folder.name.split(" - ")[2]

        split_suffix = configuration.config["split.suffix"].data

        for file in os.scandir(folder):
            with open(file, "r") as f:

                filename = os.path.splitext(file.name)[0]
                if (split and not filename.endswith(split_suffix)) or (not split and filename.endswith(split_suffix)):
                    continue
                for line in f:
                    data = line.strip()
                    self.data.append(data)

    def __load_specific(self, folder, analysis_type, split):
        self.target = folder.name.split(" - ")[2]
        filename = analysis_type.value
        filename += "_" + configuration.config["split.suffix"].data if split else ""
        filename += ".txt"

        file = os.path.join(folder, filename)
        self.data = []
        try:
            f = open(file, "r")
        except FileNotFoundError:
            print("No file " + filename + " in " + folder.name)
            return
        else:
            with f:
                for line in f:
                    data = line.strip()
                    self.data.append(data)

    def __init__(self, folder, analysis_type, split):
        if analysis_type == AnalysisType.ALL:
            self.__load_all(folder, split)
        else:
            self.__load_specific(folder, analysis_type, split)

        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))#.union(configuration.config["stopwords"].data.split(","))

        data = [i.lower() for i in self.data]
        stopped_data = [i for i in data if i not in stop_words]
        stemmed_data = [stemmer.stem(i) for i in stopped_data]

        self.data = " "
        self.data = self.data.join([w for w in stemmed_data if len(w) > 3])


class Repositories:
    def __init__(self, analysis_type, split):
        self.data = {}

        for folder in os.scandir(configuration.config["output-folder"].data):
            if os.path.isdir(folder):
                self.data[folder.name] = Repository(folder, analysis_type, split)
