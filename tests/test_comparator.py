from time import perf_counter

import pytest

from interegular import parse_pattern, Comparator

REGEX_TO_TEST = {
    "A": "a+",
    "B": "[ab]+",
    "C": "b+",
    'OP': '[+*]|[?](?![a-z])',
    'RULE_MODIFIERS': '(!|![?]?|[?]!?)(?=[_a-z])',
    'EXIT_TAG': '#(?:[ \t]+)?(?i:exit)',
    'COMMENT': '#[ \t]*(?!if|ifdef|else|elif|endif|define|set|unset|error|exit)[^\n]+|(;|//)[^\n]*'
}


@pytest.fixture
def comp(request):
    return Comparator.from_regexes({name: REGEX_TO_TEST[name] for name in request.param})


basic_collisions = [
    pytest.param(("A", "B"), (("A", "B"),), id="AB"),
    pytest.param(("A", "B", "C"), (("A", "B"), ("B", "C")), id="ABC"),
    pytest.param(("A", "C"), (), id="AC"),
    pytest.param(("OP", "RULE_MODIFIERS"), (("OP", "RULE_MODIFIERS"),), id="LOOKAHEAD"),
]


@pytest.mark.parametrize("comp, expected", basic_collisions, indirect=['comp'])
def test_check(comp, expected):
    expected = set(expected)
    for collision in comp.check():
        assert collision in expected
        expected.remove(collision)
    assert not expected


@pytest.mark.parametrize("comp, expected", basic_collisions, indirect=['comp'])
def test_example(comp, expected):
    for collision in expected:
        example = comp.get_example_overlap(*collision)
        assert comp.get_fsm(collision[0]).accepts(example.full_text), repr(example)
        assert comp.get_fsm(collision[1]).accepts(example.full_text), repr(example)


@pytest.mark.parametrize("comp, expected", [
    pytest.param(('EXIT_TAG', 'COMMENT'), (('EXIT_TAG', 'COMMENT'),), id="SLOW_EXAMPLE")
], indirect=['comp'])
def test_slow_example(comp, expected):
    for collision in expected:
        start = perf_counter()
        assert not comp.isdisjoint(*collision)
        try:
            example = comp.get_example_overlap(*collision, 0.5)
            assert comp.get_fsm(collision[0]).accepts(example)
            assert comp.get_fsm(collision[1]).accepts(example)
        except ValueError:
            pass
        end = perf_counter()
        assert end - start < 1


def test_empty():
    comp = Comparator({})
    assert comp.marked_pairs == set()
    assert comp.count_marked_pairs() == 0
    for a, b in comp.check():
        assert False, "We can't get here"
