from time import time
from ast import literal_eval

from interegular import parse_pattern, compare_patterns

start = time()
with open('terminals.pydata') as f:
    data = literal_eval(f.read())

regexes = [(n, v) for t, n, v in data if t == 're']

first = time()

patterns = {parse_pattern(v): n for n, v in regexes}

second = time()
for a, b in compare_patterns(*patterns.keys()):
    print(f"Collision between {patterns[a]} and {patterns[b]}")

end = time()

print(f"Total: {end - start}")
print(f"Data loading: {first - start}")
print(f"Regex parsing: {second - first}")
print(f"Regex comparing: {end - second}")
