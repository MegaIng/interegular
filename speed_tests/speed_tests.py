from time import time
from ast import literal_eval

from interegular import parse_pattern, compare_patterns, Comparator

start = time()
with open('terminals.pydata') as f:
    data = literal_eval(f.read())

regexes = [(n, v) for t, n, v in data if t == 're']

first = time()

patterns = {parse_pattern(v): n for n, v in regexes}

second = time()
c = Comparator({p: p for p in patterns})
for p in patterns:
    c.get_fsm(p)

third = time()
for a, b in c.check():
    print(f"Collision between {patterns[a]} and {patterns[b]}")

end = time()

print(f"Total: {end - start}")
print(f"Data loading: {first - start}")
print(f"Regex parsing: {second - first}")
print(f"FSM construction: {third - second}")
print(f"Regex comparing: {end - third}")

print(len(c._alphabet), sorted(c._alphabet))
print(len(c._alphabet.by_transition), sorted(c._alphabet.by_transition))
print(c._alphabet)
