import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


def get_ivium_path_from_folder_with_id(
    folder: str, number: int, index: int = 0, file_type: str = ".csv"
) -> str:
    fs = [
        f
        for f in os.listdir(folder)
        if (str(number) in f.split("_")[index]) and (file_type in f.split("_")[-1])
    ]
    if len(fs) > 1:
        print("Multiple Files detected")
        print(fs)
        return ""
    elif len(fs) < 1:
        print("No files detected")
        return ""
    else:
        return os.path.join(folder, fs[0])


def get_data_from_file_with_id(
    folder: str, number: int, index: int = 0, file_type: str = ".csv"
) -> pd.DataFrame:
    path = get_ivium_path_from_folder_with_id(folder, number, index, file_type)
    data = load_ivium_file(path, file_type)
    return data


def load_ivium_file(
    path: str = "", file_type: str = ".csv", cols: List[str] = ["t_s", "I_A", "E_V"]
):
    if path != "":
        if file_type == ".csv":
            return pd.read_csv(path, names=cols).astype("float64")
    print("EMPTY DATAFRAME RETURNED")
    return pd.DataFrame()


@dataclass
class IviumFile:
    path: str = field(default="")
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def set_path(self, path):
        setattr(self, "path", path)

    def set_path_from_folder_with_id(
        self, folder: str, number: int, index: int = 0, file_type: str = ".csv"
    ):
        path = get_ivium_path_from_folder_with_id(
            folder=folder, number=number, index=index, file_type=file_type
        )
        self.set_path(path)

    def read_data_from_file(self, file_type: str = ".csv"):
        setattr(self, "data", load_ivium_file(self.path, file_type))

    @property
    def t(self) -> np.ndarray:
        return self.data[self.data.columns[0]].to_numpy()

    @property
    def I(self) -> np.ndarray:
        return self.data[self.data.columns[1]].to_numpy()

    @property
    def E(self) -> np.ndarray:
        return self.data[self.data.columns[2]].to_numpy()
