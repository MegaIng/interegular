from time import time

from lark import Lark

from interegular import compare_regexes

start = time()

grammar = Lark(open("grammar.lark"), parser='lalr', start='file_input')
grammar.lexer = grammar._build_lexer()
first = time()
terminals = [(term.pattern.type, term.name, term.pattern.to_regexp()) for term in grammar.lexer.terminals]


with open('terminals.pydata', 'w') as f:
    print(repr(terminals), file=f)
second = time()

re_term_map = {re: name for type_, name, re in terminals if type_ == "re"}
for a, b in compare_regexes(*re_term_map.keys()):
    print(f"Collision between {re_term_map[a]} and {re_term_map[b]}")

end = time()
print(f"Total time: {end - start}")
print(f"Lark loading: {first - start}")
print(f"Terminal storing: {second - first}")
print(f"Comparing: {end - second}")
