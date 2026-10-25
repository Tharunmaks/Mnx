"""Remove the Python comments from a line of code.

tags: string, strip, comments, code, remove
"""

def strip_comments(line):
    return line.split("#")[0].rstrip()

# tests
assert strip_comments('x = 1  # set x') == 'x = 1'
assert strip_comments('y = 2') == 'y = 2'
