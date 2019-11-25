"""A scorer for BI[L]O[U] segmentations

Usage:
  wasp [options] <file-name>

Arguments:
  <file-name>  	The file to score (CoNLL with BI[L]O[U] labels)

Options:
  -h --help     Show this screen.
  --bio  	Use BIO mode instead of BILOU
  --gold-column <g>  	The indice of the column containing the gold labels [default: -1]
  --label-regex <r>  	A regular expression matching the labels [default: (?P<type>.*)_(?P<action>[BILOU])]
  --sys-column <s>  	The indice of the column containing the system labels [default: -2]
  --version     Show version.
"""
import pprint
import re
from typing import Iterable, List, NamedTuple, Optional, Tuple
from docopt import docopt
from wasp import __version__


class TypedSpan(NamedTuple):
    start: int
    end: int
    type: Optional[str]


def spans_from_labels(labels: Iterable[Tuple[str, Optional[str]]], bilou: bool = True):
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
) -> Tuple[int, int, int]:
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
    tru_pos = len(gold_spans.intersection(syst_spans))
    tru = len(gold_spans)
    pos = len(syst_spans)

    return tru_pos, tru, pos


def process_file(
    lines: Iterable[str],
    label_regex: str,
    gold_column: int,
    syst_column: int,
    bilou: bool,
) -> Tuple[int, int, int]:
    current_block = []
    tru_pos, tru, pos = 0, 0, 0
    for i, l in enumerate(lines, start=1):
        if not l:
            try:
                tp, t, p = process_block(
                    current_block,
                    label_regex=label_regex,
                    gold_column=gold_column,
                    syst_column=syst_column,
                    bilou=bilou,
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid value in block starting at line {i-len(current_block)}"
                ) from e
            tru_pos += tp
            tru += t
            pos += p
            current_block = []
        else:
            current_block.append(l)
    return tru_pos, tru, pos


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=f"WASp {__version__}")
    with open(arguments["<file-name>"]) as in_stream:
        tru_pos, tru, pos = process_file(
            (l.strip() for l in in_stream),
            label_regex=arguments["--label-regex"],
            gold_column=int(arguments["--gold-column"]),
            syst_column=int(arguments["--sys-column"]),
            bilou=not arguments["--bio"],
        )
    p = tru_pos / pos
    r = tru_pos / tru
    f = 2 * tru_pos / (tru + pos)
    print(f"P: {p}\tR: {r}\t F: {f}")


if __name__ == '__main__':
    main_entry_point()
