"""A scorer for BI[L]O[U] segmentations

Usage:
  wasp list
  wasp [options] <file-name>

Arguments:
  <file-name>  	The file to score (CoNLL with BI[L]O[U] labels)

Options:
  -h --help     Show this screen.
  --bio  	Use BIO mode instead of BILOU
  --gold-column <g>  	The indice of the column containing the gold labels [default: -1]
  --label-regex <r>  	A regular expression matching the labels [default: (?P<type>.*)_(?P<action>[BILOU])]
  --score <s>  	The scoring function, use `wasp list-score` to get a list [default: strict]
  --sys-column <s>  	The indice of the column containing the system labels [default: -2]
  --version     Show version.
"""
import math
import pprint
import re
from typing import Callable, Collection, Iterable, List, NamedTuple, Optional, Tuple

from docopt import docopt
import numpy as np
from scipy.optimize import linear_sum_assignment

from wasp import __version__


class TypedSpan(NamedTuple):
    start: int
    end: int
    type: Optional[str]


def exact_coef(a: TypedSpan, b: TypedSpan) -> float:
    return 1.0 if a == b else 0.0


def dice_coef(a: TypedSpan, b: TypedSpan) -> float:
    if a.type != b.type:
        return 0.0
    return (
        2
        * max(min(a.end, b.end) - max(a.start, b.start), 0)
        / (a.end - a.start + b.end - b.start)
    )


def aligned_score(
    gold: Collection[TypedSpan],
    syst: Collection[TypedSpan],
    score: Callable[[TypedSpan, TypedSpan], float],
) -> Tuple[float, float, float]:
    cost_matrix = np.array([[-score(g, s) for g in gold] for s in syst])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_score = -cost_matrix[row_ind, col_ind].sum()
    pos = math.fsum(score(s, s) for s in syst)
    tru = math.fsum(score(g, g) for g in gold)
    return total_score, pos, tru


SCORES = {
    "strict": exact_coef,
    "dice": dice_coef,
}


def spans_from_labels(labels: Iterable[Tuple[str, Optional[str]]], bilou: bool):
    """Extract a list of typed spans from labels."""
    spans: List[TypedSpan] = []
    current_start = None
    current_type = None
    for i, (label_action, label_type) in enumerate(labels):
        if label_action == "B":
            if current_start is not None:
                if bilou:
                    raise ValueError(f"Invalid label at {i}: {label_action}")
                spans.append(TypedSpan(current_start, i, current_type))
            current_start = i
            current_type = label_type
        elif label_action == "I":
            if current_start is None:
                raise ValueError(f"Invalid label action at {i}: {label_action}")
            if label_type != current_type:
                raise ValueError(f"Incoherent label type at {i}: {label_type}")
        elif label_action == "L":
            if not bilou:
                raise ValueError('Label "L" invalid in BIO mode')
            if current_start is None:
                raise ValueError(f"Invalid label at {i}: {label_action}")
            if label_type != current_type:
                raise ValueError(f"Incoherent label type at {i}: {label_action}")
            spans.append(TypedSpan(current_start, i + 1, current_type))
            current_start = None
            current_type = None
        elif label_action == "O":
            if current_start is not None:
                if bilou:
                    raise ValueError(f"Invalid label at {i}: {label_action}")
                spans.append(TypedSpan(current_start, i, current_type))
                current_start = None
                current_type = None
        elif label_action == "U":
            if not bilou:
                raise ValueError('Label "U" invalid in BIO mode')
            if current_start is not None:
                raise ValueError(f"Invalid label at {i}: {label_action}")
            spans.append(TypedSpan(i, i + 1, label_type))
    if current_start is not None:
        if bilou:
            raise ValueError("Unclosed segment")
        spans.append(TypedSpan(current_start, i + 1, current_type))
    return spans


def process_label(label: str, label_regex) -> Tuple[str, Optional[str]]:
    label_match = re.match(label_regex, label)
    if not label_match:
        raise ValueError(f"Invalid label {label!r}")
    groups = label_match.groupdict()
    try:
        return (groups["action"], groups.get("type"))
    except KeyError as e:
        raise ValueError(f"Invalid label regex: missing group {e.args[0]!r}")


def process_block(
    block: Iterable[str],
    label_regex: str,
    gold_column: int,
    syst_column: int,
    bilou: bool,
    score: Callable[[TypedSpan, TypedSpan], float],
) -> Tuple[float, float, float]:
    gold_labels = []
    syst_labels = []
    for line in block:
        columns = line.split()
        try:
            gold_labels.append(process_label(columns[gold_column], label_regex))
            syst_labels.append(process_label(columns[syst_column], label_regex))
        except ValueError as e:
            raise ValueError(f"Invalid line {line!r}") from e
    try:
        gold_spans = set(spans_from_labels(gold_labels, bilou=bilou))
    except ValueError as e:
        raise ValueError(
            f"Invalid gold label sequence:\n{pprint.pformat(list(enumerate(gold_labels)))}"
        ) from e
    try:
        syst_spans = set(spans_from_labels(syst_labels, bilou=bilou))
    except ValueError as e:
        raise ValueError(
            f"Invalid sys label sequence:\n{pprint.pformat(list(enumerate(syst_labels)))}"
        ) from e

    return aligned_score(gold_spans, syst_spans, score)


def process_file(
    lines: Iterable[str],
    label_regex: str,
    gold_column: int,
    syst_column: int,
    bilou: bool,
    score: Callable[[TypedSpan, TypedSpan], float],
) -> Tuple[float, float, float]:
    current_block: List[str] = []
    tru_pos, tru, pos = [], [], []
    for i, l in enumerate(lines, start=1):
        if not l:
            try:
                tp, t, p = process_block(
                    current_block,
                    label_regex=label_regex,
                    gold_column=gold_column,
                    syst_column=syst_column,
                    bilou=bilou,
                    score=score,
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid value in block starting at line {i-len(current_block)}"
                ) from e
            tru_pos.append(tp)
            tru.append(t)
            pos.append(p)
            current_block = []
        else:
            current_block.append(l)
    return math.fsum(tru_pos), math.fsum(tru), math.fsum(pos)


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=f"WASp {__version__}")
    if arguments["list"]:
        print("\n".join(SCORES.keys()))
        return
    if arguments["--score"] is None:
        arguments["--score"] = "strict"
    with open(arguments["<file-name>"]) as in_stream:
        tru_pos, tru, pos = process_file(
            (l.strip() for l in in_stream),
            label_regex=arguments["--label-regex"],
            gold_column=int(arguments["--gold-column"]),
            syst_column=int(arguments["--sys-column"]),
            bilou=not arguments["--bio"],
            score=SCORES[arguments["--score"]],
        )
    p = tru_pos / pos
    r = tru_pos / tru
    f = 2 * tru_pos / (tru + pos)
    print(f"P: {p}\tR: {r}\t F: {f}")


if __name__ == '__main__':
    main_entry_point()
