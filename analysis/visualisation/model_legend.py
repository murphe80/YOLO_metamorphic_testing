"""Shared numeric model ordering for RQ1 plot legends."""

import re


def sorted_model_legend(handles, labels):
    """Order legend entries by model version while retaining their plot styles."""
    entries = sorted(
        zip(handles, labels),
        key=lambda entry: (
            tuple(int(number) for number in re.findall(r"\d+", entry[1])),
            entry[1].casefold(),
        ),
    )
    return [handle for handle, _ in entries], [label for _, label in entries]
