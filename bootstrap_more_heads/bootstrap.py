"""
If one or more lhs terms are specified, all the corresponding rhs terms are gathered.
If rhs terms are specicied, there are used to.

taking this list of rhs terms ... all corresponding lhs terms are gathered.
returning {bootstrapped_lhs: {corresponding rhs: count, ...}, ...}
"""

from __future__ import \
    annotations  # ensure compatibility with cluster python version

import concurrent.futures
import itertools
import pathlib
import re
import sys
import time
import typing
from collections import Counter, defaultdict
from functools import reduce
from itertools import cycle

from pprint import pprint as pp
import ijson
import numpy as np
import orjson
import pandas as pd
import requests
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# load scripts in ./analysers
# e.g., eval callable via f = eval("evalled.script.f")
for p in pathlib.Path("bootstrappers").glob("*.py"):
    exec(f"import bootstrappers.{p.stem}")


def main(argv):

    # load the path synonynms using in configs
    # e.g., {"DATA": "~/surfdrive/data"}
    with open("path_syns.json", "rb") as f:
        path_syns = orjson.loads(f.read())

    # load the configs - prefer CL specified over default
    # [
    #   {
    #       "desc": "config 1",
    #       "switch": true,
    #       "output_dir": "DATA/project1"
    #   }
    # ]
    try:
        configs: list = get_configs(argv[0], path_syns=path_syns)
    except:
        configs: list = get_configs("bootstrap_configs.json", path_syns=path_syns)

    # iterate over configs
    for config in configs:

        desc = config["desc"]
        print(f"config={desc}")

        # get config options
        switch: bool = config["switch"]  # run config or skip?

        output_dir = resolve_fp(config["output_dir"], path_syns=path_syns)

        input_dir: pathlib.Path = resolve_fp(
            config["input_collection"][0], path_syns=path_syns
        )
        input_pattern: re.Pattern = eval(config["input_collection"][1])

        n_processes: int = config["n_processes"]

        bootstrapper: typing.Callable = eval(config["bootstrapper"])

        seed_head_patterns = config["seed_head_patterns"]
        seed_modifiers = config["seed_modifiers"]
        modifiers_ignore = config["modifiers_ignore"]

        ignore_hyphen: bool = config["ignore_hyphen"]

        wordset_fp = resolve_fp(config["wordset_fp"])

        # config to be run?
        if switch == False:
            print("\tconfig switched off ... skipping")

        else:

            fps = list(
                gen_dir(
                    input_dir,
                    pattern=input_pattern,
                    ignore_pattern=re.compile(r"config\.json"),
                )
            )

            ## open wordset of real words
            with open(wordset_fp, "r") as f:
                wordset: set = set([line.strip("\n").lower() for line in f.readlines()])

            ## bootstrap more rhs terms
            (
                modifiers_counts_by_head,
                modifiers_locs_by_head,
                doc_label2i,
            ) = bootstrapper(
                fps,
                seed_head_patterns,
                seed_modifiers,
                modifiers_ignore,
                ignore_hyphen,
                n_processes,
                output_dir,
                wordset,
            )

            # save
            save_fp = output_dir / "modifers_count_by_head.json"
            save_fp.parent.mkdir(exist_ok=True, parents=True)
            with open(save_fp, "wb") as f:
                f.write(orjson.dumps(modifiers_counts_by_head))

            save_fp = output_dir / "modifers_locs_by_head.json"
            with open(save_fp, "wb") as f:
                f.write(orjson.dumps(modifiers_locs_by_head))

            save_fp = output_dir / "doc_label2i.json"
            with open(save_fp, "wb") as f:
                f.write(orjson.dumps(doc_label2i))

        # save a copy of the config
        save_fp = output_dir / "bootstrap_more_heads_config.json"
        save_fp.parent.mkdir(exist_ok=True, parents=True)
        with open(save_fp, "wb") as f:
            f.write(orjson.dumps(config))


def resolve_fp(path: str, path_syns: typing.Union[None, dict] = None) -> pathlib.Path:
    """Resolve path synonyns, ~, and make absolute, returning pathlib.Path.

    Args:
        path (str): file path or dir
        path_syns (dict): dict of
            string to be replaced : string to do the replacing

    E.g.,
        path_syns = {"DATA": "~/documents/data"}

        resolve_fp("DATA/project/run.py")
        # >> user/home/john_smith/documents/data/project/run.py
    """

    # resolve path synonyms
    if path_syns is not None:
        for fake, real in path_syns.items():
            path = path.replace(fake, real)

    # expand user and resolve path
    return pathlib.Path(path).expanduser().resolve()


def get_configs(config_fp_str: str, *, path_syns=None) -> list:
    """Return the configs to run."""

    configs_fp = resolve_fp(config_fp_str, path_syns)

    with open(configs_fp, "rb") as f:
        configs = orjson.loads(f.read())

    return configs


def gen_dir(
    dir_path: pathlib.Path,
    *,
    pattern: re.Pattern = re.compile(".+"),
    ignore_pattern: typing.Union[re.Pattern, None] = None,
) -> typing.Generator:
    """Return a generator yielding pathlib.Path objects in a directory,
    optionally matching a pattern.

    Args:
        dir (str): directory from which to retrieve file names [default: script dir]
        pattern (re.Pattern): re.search pattern to match wanted files [default: all files]
        ignore (re.Pattern): re.search pattern to ignore wrt., previously matched files
    """

    for fp in filter(lambda fp: re.search(pattern, str(fp.name)), dir_path.glob("*")):

        # no ignore pattern specified
        if ignore_pattern is None:
            yield fp
        else:
            # ignore pattern specified, but not met
            if re.search(ignore_pattern, str(fp)):
                pass
            else:
                yield fp


if __name__ == "__main__":
    main(sys.argv[1:])  # assumes an alternative config path may be passed to CL
