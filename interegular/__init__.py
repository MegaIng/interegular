"""
A package to compare python-style regexes and test if they have intersections.
Based on the `greenery`-package by @qntm, adapted and specialized for `lark-parser`
"""

from itertools import combinations
from typing import Iterable, Tuple

from interegular.fsm import FSM
from interegular.patterns import Pattern, parse_pattern, REFlags

__all__ = ['FSM', 'Pattern', 'parse_pattern', 'compare_patterns', 'compare_regexes', '__version__']


def compare_regexes(*regexes: str) -> Iterable[Tuple[str, str]]:
    """
    Compiles the regexes to Patterns and then calls `compare_patterns` to check for intersections.
    If it finds some, returns a tuple with the two original regex-strings.
    """
    ps = {parse_pattern(r): r for r in regexes}
    yield from ((ps[a], ps[b]) for a, b in compare_patterns(*ps))


def compare_patterns(*ps: Pattern) -> Iterable[Tuple[Pattern, Pattern]]:
    """
    Compiles the Patterns to FSM and then checks them for intersections.
    If it finds some, returns a tuple with the two original Patterns.
    """
    alphabet = frozenset(c for p in ps for c in p.alphabet)
    prefix_postfix_s = [p.prefix_postfix for p in ps]
    prefix_postfix = max(p[0] for p in prefix_postfix_s), max(p[1] for p in prefix_postfix_s)
    fsms = [(p, p.to_fsm(alphabet, prefix_postfix)) for p in ps]
    for (ka, fa), (kb, fb) in combinations(fsms, 2):
        if not fa.isdisjoint(fb):
            yield (ka, kb)


__version__ = "0.1"
