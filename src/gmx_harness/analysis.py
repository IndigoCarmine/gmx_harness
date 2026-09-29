"""
This module provides tools for analyzing GROMACS simulation data.
It includes functionalities for recording and managing analysis results,
processing multiple simulation files, and generating analysis scripts.
"""

import copy
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial, total_ordering
from typing import Any, Callable, Iterator, Protocol, Self

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
import pandas as pd

from .safety import UnsafeOperationError, validate_filename, validate_name
from .scripts import GMX_SETUP, gmx_command

logger = logging.getLogger("gmx_harness")


@total_ordering
class Recorder:
    """
    A class to record and manage analysis results for a specific calculation.
    It stores named values and log data, and supports concatenation with other Recorders.
    """

    calc_name: str
    values: list[tuple[str, float]]
    log_data: list[str]

    def __init__(self, calculation_name: str):
        """
        Initializes a new Recorder instance.
        Args:
            calculation_name (str): The name of the calculation this recorder is for.
        """
        self.calc_name = calculation_name
        self.values = []
        self.log_data = []

    def log(self, val: object) -> None:
        """
        Adds a log entry to the recorder.
        Args:
            val: The value to log.
        """
        self.log_data.append(str(val))

    def add_value(self, name: str, val: float) -> None:
        """
        Adds a named numerical value to the recorder.
        Args:
            name (str): The name of the value.
            val (float): The numerical value.
        """
        self.values.append((name, val))

    def add_value_of_default_array_analysis(self, name: str, array: npt.NDArray[Any]) -> None:
        """
        Adds mean and standard deviation of a numerical array as named values.
        Args:
            name (str): The base name for the values (e.g., "energy" will result in "energy_mean" and "energy_standard").
            array (float): The numerical array to analyze.
        """
        self.add_value(f"{name}_mean", float(np.mean(array)))
        self.add_value(f"{name}_standard", float(np.std(array)))

    def get_all_valuename(self) -> list[str]:
        """
        Returns a list of all value names recorded.
        Returns:
            list[str]: A list of value names.
        """
        return [name for name, _ in self.values]

    def concat(self, recorder: "Recorder") -> None:
        """
        Concatenates another Recorder's values into this recorder.
        Raises ValueError if calculation names do not match or if there are duplicate value names.
        Args:
            recorder (Recorder): The Recorder object to concatenate.
        """
        if self.calc_name != recorder.calc_name:
            raise ValueError(
                "Recorders dont have a same calculation_name: self is {} but arg is {}".format(
                    self.calc_name, recorder.calc_name
                )
            )

        value_name = [val for val, _ in self.values]
        for name, val in recorder.values:
            if name in value_name:
                raise ValueError(f"Recorders have a same value:{name}")
            else:
                self.values.append((name, val))

    def __eq__(self, other: object) -> bool:
        """
        Compares two Recorder objects for equality based on their calculation name.
        """
        if not isinstance(other, Recorder):
            raise TypeError("Cannot compare with Recorder and {}".format(type(other)))

        return self.calc_name == other.calc_name

    def __lt__(self, other: object) -> bool:
        """
        Compares two Recorder objects for less than based on their calculation name.
        """
        if not isinstance(other, Recorder):
            raise TypeError("Cannot compare with Recorder and {}".format(type(other)))

        return self.calc_name < other.calc_name


def generate_excel(file_name: str, recoders: list[Recorder]) -> None:
    """
    Generates an Excel file from a list of Recorder objects.
    Each Recorder's values are written as a row in the Excel file.
    Args:
        file_name (str): The name of the Excel file to generate.
        recoders (list[Recorder]): A list of Recorder objects containing the data.
    """
    # all = set()
    # for recoder in recoders:
    #     for val in recoder:
    #         all.add(val)
    if not recoders:
        raise ValueError("recoders is empty")
    all = [name for name, _ in recoders[0].values]
    # all = ["name"] + all
    data = pd.DataFrame()

    for recoder in recoders:
        data[recoder.calc_name] = [val for _, val in recoder.values]
    logger.debug("%s", data)
    data = data.T
    data.columns = all
    data.to_excel(file_name)


def mixing_recorders(
    base_recorder_list: list[Recorder], recorder_list: list[Recorder]
) -> list[Recorder]:
    """
    Mixes (concatenates) values from a list of new recorders into a base list of recorders.
    Recorders are matched by their `calc_name`.
    Args:
        base_recorder_list (list[Recorder]): The base list of recorders to which values will be added.
        recorder_list (list[Recorder]): The list of recorders whose values will be added.
    Returns:
        list[Recorder]: A new list of recorders with mixed values.
    """
    new_recorders = copy.deepcopy(base_recorder_list)

    for recorder in new_recorders:
        for add_recorder in recorder_list:
            if recorder.calc_name == add_recorder.calc_name:
                recorder.concat(add_recorder)

    return new_recorders


def process_files(
    calculation_basedir: str,
    last_calc: str,
    work: Callable[[str], Recorder | None],
    do_parallel: bool = False,
) -> list[Recorder]:
    """
    Processes files within a specified calculation base directory.
    Applies a given work function to each relevant directory, optionally in parallel.
    Args:
        calculation_basedir (str): The base directory containing calculation folders.
        last_calc: The name of the last calculation subdirectory (e.g., "4_md_main").
        work (Callable[[str], Recorder | None]): A function that takes a directory path and returns a Recorder object or None.
        do_parallel (bool): If True, processes files in parallel using a ProcessPoolExecutor.
    Returns:
        list[Recorder]: A list of Recorder objects generated by the work function.
    """

    # get fullpathes of all file and dir in calculation_basedir
    directry = [
        os.path.join(calculation_basedir, dirname)
        for dirname in os.listdir(calculation_basedir)
    ]

    # convert to pathes of last_calc dir.
    directry = [dir for dir in directry if os.path.isdir(dir)]
    directry = [os.path.join(dir, last_calc) for dir in directry]
    directry = [dir for dir in directry if os.path.exists(dir)]
    recorders: list[Recorder] = []
    if do_parallel:
        with ProcessPoolExecutor() as executor:
            result = executor.map(work, directry)
            for recorder in result:
                if recorder is not None:
                    recorders.append(recorder)

    else:
        for dir in directry:
            recorder = work(dir)
            if recorder is not None:
                recorders.append(recorder)

    return recorders


def _analyze_trj_inner(path: str, work: Callable[[Any, str], Recorder | None]) -> Recorder | None:
    """
    Internal helper function to analyze a single trajectory file.
    Loads the GROMACS universe and applies a work function.
    Args:
        path (str): The path to the directory containing output.gro and output.xtc.
        work (Callable[[mda.Universe, str], Recorder]): A function that takes an MDAnalysis Universe object and the path, and returns a Recorder.
    Returns:
        Recorder | None: A Recorder object with analysis results, or None if files are not found.
    """
    grofile = os.path.join(path, "output.gro")
    xtcfile = os.path.join(path, "output.xtc")
    if not os.path.exists(grofile):
        warnings.warn(f"{grofile} does not exist", stacklevel=2)
        return None
    u = mda.Universe(grofile)
    if os.path.exists(xtcfile):
        u.load_new(xtcfile)
    return work(u, path)


def analyze_trj(
    calculation_basedir: str,
    last_calc: str,
    work: Callable[[Any, str], Recorder | None],
    do_parallel: bool = False,
) -> list[Recorder]:
    """
    Analyzes GROMACS trajectory files across multiple calculation directories.
    It uses `process_files` internally to handle parallel processing.
    Args:
        calculation_basedir (str): The base directory containing calculation folders.
        last_calc: The name of the last calculation subdirectory (e.g., "4_md_main").
        work (Callable[[mda.Universe, str], Recorder | None]): A function that takes an MDAnalysis Universe object and the path, and returns a Recorder.
        do_parallel (bool): If True, processes files in parallel.
    Returns:
        list[Recorder]: A list of Recorder objects with analysis results.
    """
    return process_files(
        calculation_basedir,
        last_calc,
        partial(_analyze_trj_inner, work=work),
        do_parallel,
    )


def get_calcname(calc_path: str) -> str:
    """
    Extracts the calculation name from a given calculation path.
    Example: "...../iQuin/4_md_main" -> "iQuin"
    Args:
        calc_path (str): The full path to the calculation directory.
    Returns:
        str: The extracted calculation name.
    """

    splitedpath = os.path.split(calc_path)

    if not splitedpath[1][0].isdigit():
        warnings.warn(f"{calc_path} may not be calculation path.", stacklevel=2)

    return os.path.split(splitedpath[0])[1]


def grouping[T](groups: list[T], group_size: int) -> Iterator[list[T]]:
    """
    Groups a list into sub-lists of a specified size.
    Args:
        groups (list[T]): The list to be grouped.
        group_size (int): The desired size of each sub-group.
    Returns:
        Iterator[list[T]]: An iterator yielding sub-lists.

    Example:
    for group in grouping([1,2,3,4,5,6,7,8,9], 3):
        print(group)
        # [[1, 2, 3],
        # [4, 5, 6],
        # [7, 8, 9]]
    """
    for i in range(0, len(groups), group_size):
        yield groups[i : i + group_size]


class _Addable(Protocol):
    """
    A Protocol defining types that support the addition operation.
    """

    def __add__(self, other: Self, /) -> Self: ...


def concat[A: _Addable](group: list[A]) -> A:
    """
    Concatenates a list of addable objects using their __add__ method.
    Args:
        group (list[_Addable]): A list of objects that support addition.
    Returns:
        _Addable: The concatenated result.
    """
    temp = group[0]
    for i in range(1, len(group)):
        temp = temp + group[i]
    return temp


def make_analyze_command_script(
    basedir: str,
    calcname: str,
    command: str | list[str],
    commandname: str,
    *,
    stdin: str | None = None,
    allow_unsafe: bool = False,
) -> str:
    """
    Write ``basedir/<commandname>.sh`` that runs one analysis command in every
    ``basedir/*/<calcname>`` directory. Returns the script path.

    ``command`` is either a gmx argv (safe, quoted), e.g.
    ``["hbond", "-f", "output.xtc", "-s", "output.tpr", "-num", "hbond.xvg"]`` with
    ``stdin="0 0"`` (group selections), or a raw shell string, which is unchecked
    and needs ``allow_unsafe=True``. ``"$GMX"`` is the GROMACS command in the script.
    """
    validate_name(calcname, "calcname")
    validate_filename(commandname + ".sh", "commandname")
    if isinstance(command, str):
        if not allow_unsafe:
            raise UnsafeOperationError(
                "a raw command string is written unchecked; pass a gmx argv list, or allow_unsafe=True"
            )
        line = command
    else:
        line = gmx_command(command[0], list(command[1:]), stdin=stdin)

    script = ["#!/bin/bash", "set -e", GMX_SETUP]
    for folder in sorted(os.listdir(basedir)):
        if not os.path.isdir(os.path.join(basedir, folder, calcname)):
            continue
        validate_name(folder, "structure directory")
        script += [f"(cd {folder}/{calcname} && {line})", ""]
    path = os.path.join(basedir, f"{commandname}.sh")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(script))
    return path
