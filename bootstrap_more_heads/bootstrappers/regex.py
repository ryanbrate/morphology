from __future__ import \
    annotations  # ensure compatibility with cluster python version

import operator
import pathlib
import re
import typing
from collections import Counter, defaultdict
from functools import reduce
from itertools import chain, compress, cycle

from pprint import pprint as pp
import ijson
import numpy as np
import orjson
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import regex


def bootstrap(
    fps: list[pathlib.Path],
    seed_centre_patterns: list[str],
    seed_contexts: list[str],
    contexts_ignore: list[
        str
    ],  # only applies to bootstrapped heads, not the modifiers subsequently profiled for bootstrapped heads
    ignore_hyphen: bool,
    n_processes: int,
    output_dir: pathlib.Path,
    word_set: set,  # lowercased list of words considered true words
):
    """
    Return 3 dicts ...

    1)
    {
        head: {
            corresponding modifier: count
            ...
        },
        ...
    }

    2a)
    {
        head: {
            corresponding modifier: list of doc_label indices the modifier+context
            ...
        },
        ...
    }

    2b)
    {
        i: doc_label
    }
    """

    modifiers_count_by_head = defaultdict(Counter)
    modifiers_location_by_head = defaultdict(lambda: defaultdict(list))
    doc_label2i = {}

    ## 1) mine new contexts coincident with passed seed_centre_patterns, and combine with seed_contexts
    if len(seed_centre_patterns) > 0:

        # build pattern for extracting modifiers
        trie = Trie()
        for pattern in seed_centre_patterns:
            trie.add(pattern)
        if ignore_hyphen == True:
            p = "(.+?)-*" + trie.pattern()
        else:
            p = "(.+?)" + trie.pattern()

        print(f"\tgetting modifiers corresponding to seed head patterns")
        unmined_modifiers: set[str] = set(seed_contexts).union(
            *process_map(
                get_modifiers_star,
                zip(
                    fps,
                    cycle([p]),
                    range(len(fps)),
                ),
                max_workers=n_processes,
            )
        )

    else:
        unmined_modifiers = set(seed_contexts)

    ## 2) filter the modifiers according to general rules
    print("\tfilter modifier list")
    kept_modifiers: set = {
        modifier
        for modifier in unmined_modifiers
        if wanted_modifier(modifier, word_set)
    }
    # remove term in ignore list
    kept_modifiers = {
        modifier for modifier in kept_modifiers if modifier not in contexts_ignore
    }

    save_fp = output_dir / "modifiers_kept.txt"
    save_fp.parent.mkdir(exist_ok=True, parents=True)
    with open(save_fp, "w") as f:
        f.writelines([modifier + "\n" for modifier in kept_modifiers])

    save_fp = output_dir / "modifiers_discarded.txt"
    save_fp.parent.mkdir(exist_ok=True, parents=True)
    with open(save_fp, "w") as f:
        f.writelines(
            [modifier + "\n" for modifier in unmined_modifiers - kept_modifiers]
        )

    ## 3) mine new heads, which are coincident with filtered modifiers
    if len(kept_modifiers) > 0:

        trie = Trie()
        for modifier in kept_modifiers:
            trie.add(f"{re.escape(modifier)}")
        if ignore_hyphen:
            p = trie.pattern() + "-*(.+)"
        else:
            p = trie.pattern() + "(?!-)(.+)"

        # get all heads which match against unmined modifiers
        print(f"\tgetting heads corresponding to modifiers")
        bootstrapped_heads: set = set.union(
            *process_map(
                get_heads_star,
                zip(
                    fps,
                    cycle([p]),
                    range(len(fps)),
                ),
                max_workers=n_processes,
            )
        )

        # filter minded heads
        print("\tfilter bootstrapped heads")
        kept_heads: set = {
            head for head in bootstrapped_heads if wanted_head(head, word_set)
        }
        print(f"\t\t{len(kept_heads)} head kept")

        with open(output_dir / "bootstrapped_kept_heads.txt", "w") as f:
            f.writelines([head + "\n" for head in kept_heads])

        with open(output_dir / "bootstrapped__discarded_heads.txt", "w") as f:
            f.writelines([head + "\n" for head in bootstrapped_heads - kept_heads])

        # 4)
        print(f"\tget dicts")

        trie = Trie()
        for head in kept_heads:
            trie.add(f"{re.escape(head)}")
        if ignore_hyphen:
            p = "(.+?)-*" + f"({trie.pattern()})"
        else:
            p = "(.+)" + f"({trie.pattern()})"

        # process map splits creates splits of modifiers_count_by_head ... etc,
        # according to fps splits
        (
            modifiers_count_by_head_iter,  # [{head: {modifier: count, ...}, ..}, ...]
            modifiers_location_by_head_iter,  # [{head: {modifier: ["http://kb...", ...], ...}, ...}, ...]
            doc_label2i_iter,  # [{"http://kb...": 0, ...}, ...]
        ) = list(
            zip(
                *process_map(
                    get_modifiers_count_by_head_star,
                    zip(
                        fps,
                        cycle([p]),
                        cycle([word_set]),
                        range(len(fps)),
                    ),
                    max_workers=n_processes,
                    chunksize=1,
                )
            )
        )

        print(f"\tstitch the dicts")

        # join the modifiers_count_by_head splits
        modifiers_count_by_head = defaultdict(Counter)
        for modifiers_count_by_head_split in modifiers_count_by_head_iter:
            for head, modifiers_counter in modifiers_count_by_head_split.items():
                for modifier, count in modifiers_counter.items():
                    modifiers_count_by_head[head][modifier] += count

        # join the modifiers_count_by_head splits
        modifiers_location_by_head = defaultdict(lambda: defaultdict(list))
        doc_label2i = {}
        for j, (modifiers_location_by_head_split, doc_label2i_split) in enumerate(
            zip(modifiers_location_by_head_iter, doc_label2i_iter)
        ):

            i2doc_label = {i: doc_label for doc_label, i in doc_label2i_split.items()}

            for head, modifiers_loc in modifiers_location_by_head_split.items():
                for modifier, doc_indices in modifiers_loc.items():
                    for i in doc_indices:

                        doc_label = i2doc_label[i]

                        # reassign i
                        new_i = j + i

                        # update
                        doc_label2i[doc_label] = new_i
                        modifiers_location_by_head[head][modifier].append(new_i)

    return (modifiers_count_by_head, modifiers_location_by_head, doc_label2i)


def get_modifiers_count_by_head_star(t):
    return get_modifier_count_by_head(*t)


def get_modifier_count_by_head(
    fp: pathlib.Path,
    pattern: str,
    word_set: set,
    pbar_position: int,
):
    """Return tuple of dicts (modifiers_count_by_head, modifiers_location_by_head, doc_label2i)"""

    modifiers_count_by_head = defaultdict(Counter)
    modifiers_location_by_head = defaultdict(lambda: defaultdict(list))
    doc_label2i = T2i()

    fp_items = gen_items([fp])
    for item in tqdm(fp_items, desc=str(pbar_position), position=pbar_position):

        doc_label, doc_stuff = item

        for sentence_text, sentence_parse in doc_stuff:

            # iterate over each token in parse
            for parse_token in sentence_parse:

                if parse_token["pos"] == "NOUN":

                    noun = parse_token["text"]
                    match = re.match(pattern, noun)
                    if match:
                        modifier = match.groups()[0]
                        head = match.groups()[1]
                        if wanted_modifier(modifier, word_set):
                            modifiers_count_by_head[head][modifier] += 1

                            #
                            doc_label2i.append(doc_label)
                            modifiers_location_by_head[head][modifier].append(
                                doc_label2i[doc_label]
                            )

    return (
        dict(modifiers_count_by_head),
        dict(modifiers_location_by_head),
        doc_label2i.t2i,
    )


def wanted_modifier(lhs_term, word_set: set) -> bool:

    if len(lhs_term) < 3:  # ignore modifiers less than 3 chars
        return False

    # check modifier, ignoring hyphen always, is in the word_set
    if lhs_term[-1] != "-":
        if lhs_term not in word_set:
            return False
    else:
        if lhs_term[:-1] not in word_set:
            return False
    return True


def get_modifiers_star(t) -> set[str]:
    return get_modifiers(*t)


def get_modifiers(
    fp: pathlib.Path,
    seed_pattern: str,
    pbar_position: int,
) -> set[str]:
    """Return a set of modifiers co-occurrent with passed head_patterns, for pos==NOUN

    If ignore_hyphen == True, then lhs terms are reported without
    terminating hyphens
    """

    returned = set()

    # iterate over fp items
    fp_items = gen_items([fp])
    for item in tqdm(fp_items, desc=str(pbar_position), position=pbar_position):

        doc_label, doc_stuff = item

        for sentence_text, sentence_parse in doc_stuff:

            # iterate over each token in parse
            for parse_token in sentence_parse:

                if parse_token["pos"] == "NOUN":

                    noun = parse_token["text"]

                    match = re.match(seed_pattern, noun)
                    if match:
                        modifier = match.groups()[0]
                        returned.add(modifier)

    return returned


def wanted_head(rhs_term, word_set) -> bool:
    """Return True is rhs wanted."""

    # ignore heads less than 3 chars
    if len(rhs_term) < 3:
        return False

    # check lhs ignoring hyphen always is in the word_set
    if rhs_term[-1] != "-":
        if rhs_term not in word_set:
            return False
    else:
        if rhs_term[:-1] not in word_set:
            return False

    return True


def get_heads_star(t):
    return get_heads(*t)


def get_heads(fp: pathlib.Path, pattern: str, pbar_position: int) -> set:
    """Return a set of heads, for corresponnding modifiers

    If ignore_hypen == True, terminating hyphens of any lhs terms are ignored.
    """

    heads = set()

    fp_items = gen_items([fp])
    for item in tqdm(fp_items, desc=str(pbar_position), position=pbar_position):

        doc_label, doc_stuff = item

        for sentence_text, sentence_parse in doc_stuff:

            # iterate over each token in parse
            for parse_token in sentence_parse:

                if parse_token["pos"] == "NOUN":

                    noun = parse_token["text"]

                    match = re.match(pattern, noun)
                    if match:
                        head = match.groups()[0]
                        heads.add(head)

    return heads


def gen_items(fps: typing.Iterable[pathlib.Path]) -> typing.Generator:
    """Return each item (in each fp of passed fps)."""

    for fp in fps:
        with open(fp, "rb") as f:
            for item in ijson.items(f, "item"):
                    yield item


class T2i:
    """An basic index class"""

    def __init__(self):
        self.t2i = {}
        self.maxi = -1

    def __getitem__(self, t):
        return self.t2i[t]

    def append(self, t):
        if t not in self.t2i.keys():
            self.maxi += 1
            self.t2i[t] = self.maxi

    def __iter__(self):
        for x in self.t2i.items():
            yield x


class Trie:
    # from https://stackoverflow.com/questions/42742810/speed-up-millions-of-regex-replacements-in-python-3

    def __init__(self):
        self.data = {}

    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def dump(self):
        return self.data

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(
            data.keys(), key=lambda x: len(data[x].keys()), reverse=True
        ):  # sort by most children
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(char + recurse)
                except:
                    cc.append(char)
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append("[" + "".join(cc) + "]")

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self) -> str:
        return self._pattern(self.dump())
