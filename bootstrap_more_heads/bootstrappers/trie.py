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
import regex
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def bootstrap(
    fps: list[pathlib.Path],
    seed_centres: list[str],
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
    modifiers_loc_by_head = defaultdict(lambda: defaultdict(list))
    loc2i = {}

    ## 1) mine new contexts coincident with passed seed_centre_patterns, and combine with seed_contexts
    if len(seed_centres) > 0:

        print(f"\tgetting modifiers corresponding to seed head patterns")
        unmined_modifiers: set[str] = set(seed_contexts).union(
            *process_map(
                get_modifiers_star,
                zip(
                    fps,
                    cycle([seed_centres]),
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

        # get all heads which match against unmined modifiers
        print(f"\tgetting heads corresponding to modifiers")
        bootstrapped_heads: set = set(seed_centres).union(
            *process_map(
                get_heads_star,
                zip(
                    fps,
                    cycle([kept_modifiers]),
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

        # 4)  get head->modifier->loc data for set of original seed heads, plus bootstrapped heads
        print(f"\tget dicts")

        (
            modifiers_loc_by_head_iter,  # [{head: {modifier: [0, ...], ...}, ...}, ...]
            loc2i_iter,  # [{"http://kb...": 0, ...}, ...]
        ) = list(
            zip(
                *process_map(
                    get_modifiers_by_head_star,
                    zip(
                        fps,
                        cycle([kept_heads]),
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
        modifiers_count_by_head = defaultdict(lambda: defaultdict(int))
        modifiers_loc_by_head = defaultdict(lambda: defaultdict(list))
        loc2i = {}
        for j, (modifiers_loc_by_head_split, loc2i_split) in enumerate(
            zip(modifiers_loc_by_head_iter, loc2i_iter)
        ):

            i2loc_split = {i: doc_label for doc_label, i in loc2i_split.items()}

            for head, modifiers_loc in modifiers_loc_by_head_split.items():
                for modifier, loc_indices in modifiers_loc.items():

                    for i in loc_indices:

                        doc_label = i2loc_split[i]

                        # reassign i
                        new_i = j + i

                        # update
                        loc2i[doc_label] = new_i
                        modifiers_loc_by_head[head][modifier].append(new_i)

        # build modifiers_count_by_head
        print("building modifiers_count_by_head")
        for head in modifiers_loc_by_head.keys():
            for modifier, locations in modifiers_loc_by_head[head].items():
                modifiers_count_by_head[head][modifier] += len(locations)

    return (modifiers_count_by_head, modifiers_loc_by_head, loc2i)


def get_modifiers_by_head_star(t):
    return get_modifiers_by_head(*t)


def get_modifiers_by_head(
    fp: pathlib.Path,
    seed_heads: list[str],
    word_set: set,
    pbar_position: int,
):
    """Return tuple of dicts (modifiers_location_by_head, doc_label2i)"""

    modifiers_location_by_head = defaultdict(lambda: defaultdict(list))
    doc_label2i = T2i()

    trie = RevTrie()

    # build Trie capable of extracting modifiers
    doc_label2i = T2i()
    noun2i = defaultdict(list)

    fp_items = gen_items([fp])
    for item in tqdm(fp_items, desc=str(pbar_position), position=pbar_position):

        doc_label, doc_stuff = item
        doc_label2i.append(doc_label)
        i = doc_label2i[doc_label]

        for sentence_text, sentence_parse in doc_stuff:

            # iterate over each token in parse
            for parse_token in sentence_parse:

                if parse_token["pos"] == "NOUN":

                    noun = parse_token["text"]
                    trie.add(noun)

                    noun2i[noun].append(i)

    # extract and filter the modifiers for each seed head
    for seed_head in seed_heads:
        for modifier in trie.get_prefixes(seed_head):
            if wanted_modifier(modifier, word_set):
                noun = modifier + seed_head
                modifiers_location_by_head[seed_head][modifier] += noun2i[noun]

    return (
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
    seed_heads: list[str],
    pbar_position: int,
) -> set[str]:
    """Return a set of modifiers co-occurrent with passed head_patterns, for pos==NOUN

    If ignore_hyphen == True, then lhs terms are reported without
    terminating hyphens
    """

    rev_trie = RevTrie()
    # i.e., trie of words stored in reverse ...

    # build a reverse trie
    fp_items = gen_items([fp])
    for item in tqdm(fp_items, desc=str(pbar_position), position=pbar_position):

        doc_label, doc_stuff = item

        for sentence_text, sentence_parse in doc_stuff:

            # iterate over each token in parse
            for parse_token in sentence_parse:

                if parse_token["pos"] == "NOUN":

                    noun = parse_token["text"]
                    rev_trie.add(noun)

    # get the modifiers from the rev trie
    modifiers = set()
    for seed_head in seed_heads:
        modifiers = modifiers.union(rev_trie.get_prefixes(seed_head))

    return set(modifiers)


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


def get_heads(fp: pathlib.Path, seed_modifiers: str, pbar_position: int) -> set:
    """Return a set of heads, for corresponnding modifiers

    If ignore_hypen == True, terminating hyphens of any lhs terms are ignored.
    """

    trie = Trie()

    # build trie
    fp_items = gen_items([fp])
    for item in tqdm(fp_items, desc=str(pbar_position), position=pbar_position):

        doc_label, doc_stuff = item

        for sentence_text, sentence_parse in doc_stuff:

            # iterate over each token in parse
            for parse_token in sentence_parse:

                if parse_token["pos"] == "NOUN":

                    noun = parse_token["text"]
                    trie.add(noun)

    # get heads
    heads = set()
    for seed_modifer in seed_modifiers:
        heads = heads.union(trie.get_affixes(seed_modifer))

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
    """Build a trie and get affixes for some given prefix.
    
    E.g., for getting heads for some given modifiers.
    """

    def __init__(self):
        self.root = dict()

    def add(self, word: str):
        """Add word to trie"""
        ref = self.root
        for char in word:
            ref.setdefault(char, {})
            ref = ref[char]

    def get_affixes(self, prefix: str) -> list:
        """Return a list of all affixes, for given prefix."""

        # burn through the prefix ... ending with ref on the whatever comes after the prefix
        ref = self.root
        for char in prefix:
            if char in ref:
                ref = ref[char]

        acc = []
        stack = [("", ref)]

        # collect suffices
        while stack:
            collected, ref = stack.pop()

            # ref is exhausted, add collected to accumulator
            if ref == {}:
                if collected != "":
                    acc.append(collected)
                else:
                    pass

            # still more to collect, add to stack
            else:
                for char in ref.keys():
                    stack.append((collected + char, ref[char]))

        return acc


class RevTrie:
    """Build a trie (or words in reverse) and get prefixes for some given affix.

    E.g., for getting modifiers for some given heads
    """

    def __init__(self):
        self.root = dict()

    def add(self, word: str):
        """Add (reversed) word to trie.
        
            Args:
                word (str): word in original order
        """
        ref = self.root
        for char in word[::-1]:
            ref.setdefault(char, {})
            ref = ref[char]

    def get_prefixes(self, affix: str) -> list:
        """Return a list of all affixes, for given prefix."""

        # burn through the suffix ... ending with ref on the whatever comes after the suffix
        ref = self.root
        for char in affix[::-1]:
            if char in ref:
                ref = ref[char]

        acc = []
        stack = [("", ref)]

        # collect suffices
        while stack:
            collected, ref = stack.pop()

            # ref is exhausted, add collected to accumulator
            if ref == {}:
                if collected != "":
                    acc.append(collected)
                else:
                    pass

            # still more to collect, add to stack
            else:
                for char in ref.keys():
                    stack.append((collected + char, ref[char]))

        return [reversed_prefix[::-1] for reversed_prefix in acc]
