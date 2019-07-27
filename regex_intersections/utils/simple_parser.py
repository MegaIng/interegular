"""
A small util to simplify the creation of Parsers for simple context-free-grammars.

"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Generic, TypeVar, Optional, List

__all__ = ['nomatch', 'NoMatch', 'SimpleParser']


class nomatch(BaseException):
    def __init__(self):
        pass


class NoMatch(ValueError):
    def __init__(self, data: str, index: int, expected: List[str]):
        self.data = data
        self.index = index
        self.expected = expected
        super(NoMatch, self).__init__(f"Can not match at index {index}. Got {data[index:index + 5]!r},"
                                      f" expected any of {expected}.\n"
                                      f"Context(data[-10:+10]): {data[index - 10: index + 10]!r}")


T = TypeVar('T')


class SimpleParser(Generic[T], ABC):
    def __init__(self, data: str):
        self.data = data
        self.index = 0
        self._expected = defaultdict(list)

    def parse(self) -> T:
        try:
            result = self.start()
        except nomatch:
            raise NoMatch(self.data, max(self._expected), self._expected[max(self._expected)]) from None
        if self.index < len(self.data):
            raise NoMatch(self.data, max(self._expected), self._expected[max(self._expected)])
        return result

    @abstractmethod
    def start(self) -> T:
        raise NotImplementedError

    def static(self, expected: str):
        length = len(expected)
        if self.data[self.index:self.index + length] == expected:
            self.index += length
        else:
            self._expected[self.index].append(expected)
            raise nomatch

    def static_b(self, expected: str) -> bool:
        l = len(expected)
        if self.data[self.index:self.index + l] == expected:
            self.index += l
            return True
        else:
            self._expected[self.index].append(expected)
            return False

    def anyof(self, *strings: str) -> str:
        for s in strings:
            if self.static_b(s):
                return s
        else:
            raise nomatch

    def any(self, length: int = 1) -> str:
        if self.index + length <= len(self.data):
            res = self.data[self.index:self.index + length]
            self.index += length
            return res
        else:
            self._expected[self.index].append(f"<Any {length}>")
            raise nomatch

    def any_but(self, *strings, length: int = 1) -> str:
        if self.index + length <= len(self.data):
            res = self.data[self.index:self.index + length]
            if res not in strings:
                self.index += length
                return res
            else:
                self._expected[self.index].append(f"<Any {length} except {strings}>")
                raise nomatch
        else:
            self._expected[self.index].append(f"<Any {length} except {strings}>")
            raise nomatch

    def multiple(self, chars: str, mi: int, ma: Optional[int]) -> str:
        result = []
        for off in range(mi):
            if self.data[self.index + off] in chars:
                result.append(self.data[self.index + off])
            else:
                self._expected[self.index + off].extend(chars)
                raise nomatch
        self.index += mi
        if ma is None:
            while True:
                if self.data[self.index] in chars:
                    result.append(self.data[self.index])
                    self.index += 1
                else:
                    self._expected[self.index].extend(chars)
                    break
        else:
            for _ in range(ma - mi):
                if self.data[self.index] in chars:
                    result.append(self.data[self.index])
                    self.index += 1
                else:
                    self._expected[self.index].extend(chars)
                    break
        return ''.join(result)
