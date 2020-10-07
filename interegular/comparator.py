from itertools import combinations
from typing import List, Tuple, Any, Dict, Iterable, Set

from interegular.fsm import FSM
from interegular.patterns import Pattern, Unsupported, parse_pattern
from interegular.utils import logger


class Comparator:
    def __init__(self, patterns: Dict[Any, Pattern]):
        self._patterns = patterns
        if not patterns:  # `isdisjoint` can not be called anyway, so we don't need to create a valid state
            return
        self._alphabet = frozenset.union(*(p.alphabet for p in patterns.values()))
        prefix_postfix_s = [p.prefix_postfix for p in patterns.values()]
        self._prefix_postfix = max(p[0] for p in prefix_postfix_s), max(p[1] for p in prefix_postfix_s)
        self._fsms: Dict[Any, FSM] = {}
        self._know_pairs: Dict[Tuple[Any, Any], bool] = {}
        self._marked_pairs: Set[Tuple[Any, Any]] = set()

    def get_fsm(self, a: Any) -> FSM:
        if a not in self._fsms:
            try:
                self._fsms[a] = self._patterns[a].to_fsm(self._alphabet, self._prefix_postfix)
            except Unsupported as e:
                self._fsms[a] = None
                logger.warning(f"Can't compile Pattern to fsm for {a}\n     {repr(e)}")
            except KeyError:
                self._fsms[a] = None # In case it was thrown away in `from_regexes`
        return self._fsms[a]

    def isdisjoint(self, a: Any, b: Any) -> bool:
        if (a, b) not in self._know_pairs:
            fa, fb = self.get_fsm(a), self.get_fsm(b)
            if fa is None or fb is None:
                self._know_pairs[a, b] = True  # We can't know. Assume they are disjoint
            else:
                self._know_pairs[a, b] = fa.isdisjoint(fb)
        return self._know_pairs[a, b]

    def check(self, keys: Iterable[Any]) -> Iterable[Tuple[Any, Any]]:
        for a, b in combinations(keys, 2):
            if not self.isdisjoint(a, b):
                yield a, b

    def is_marked(self, a: Any, b: Any) -> bool:
        return (a, b) in self._marked_pairs

    def mark(self, a: Any, b: Any):
        self._marked_pairs.add((a, b))

    @classmethod
    def from_regexes(cls, regexes: Dict[Any, str]):
        patterns = {}
        for k, r in regexes.items():
            try:
                patterns[k] = parse_pattern(r)
            except Unsupported as e:
                logger.warning(f"Can't compile regex to Pattern for {k}\n     {repr(e)}")
        return cls(patterns)
