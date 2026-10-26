"""Turn a title into a URL-friendly slug.

tags: string, slug, url, title, format
"""

def slugify(s):
    out = "".join(c.lower() if c.isalnum() else "-" for c in s)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")

# tests
assert slugify('Hello World!') == 'hello-world'
assert slugify('') == ''
