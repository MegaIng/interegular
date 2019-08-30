# Interegular
***regex intersection checker***

A library to check a subset of python regexes for intersections.
Based on [grennery](https://github.com/qntm/greenery) by [@qntm](https://github.com/qntm). Adapted for [lark-parser](https://github.com/lark-parser/lark)

## Interface

| Function | Usage |
| -------- | ----- |
| `interegular.compare_regexes(*regexes: str)` | Takes a series of regexes as strings and returns a Generator of all intersections as `(str, str)`|
| `interegular.parse_pattern(regex: str)` | Parses a regex as string to a `Pattern` object|
| `interegular.compare_patterns(*patterns: Pattern)` | Takes a series of regexes as patterns and returns a Generator of all intersections as `(str, str)`|
| `interegular.Pattern` | A class representing a parsed regex (intermediate representation)|
| `interegular.REFlags` | A enum representing the flags a regex can have |
| `interegular.FSM` | A class representing a fully parsed regex. (Has many useful members) |
| `Pattern.with_flags(added: REFlags, removed: REFlags)` | A function to change the flags that are applied to a regex|
| `Pattern.to_fsm() -> FSM` | A function to create a `FSM` object from the Pattern |


## What is supported?

Most normal python-regex syntax is support. But because of the backend that is used (final-state-machines), some things can not be implemented. This includes:

- Backwards references (`\1`, `(?P=open)`)
- Conditional Matching (`(?(1)a|b)`)
- Some cases of lookaheads/lookbacks (You gotta try out which work and which don't)
  - A word of warning: This is currently not correctly handled, and some things might parse, but not work correctly. I am currently working on this.


Some things are simply not implemented and will implemented in the future:
- Some flags (Progress: `ims` from `aiLmsux`)
- Some cases of lookaheads/lookbacks (You gotta try out which work and which don't)