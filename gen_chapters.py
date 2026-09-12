#!/usr/bin/env python3
"""Generate all chapter files for AI Architect textbook."""
import os

BASE = r'C:\Users\SCSM11\Desktop\AI-Architect-Textbook\src'

def w(lang, part, content):
    path = os.path.join(BASE, lang, 'chapters', f'chapter-{part}.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    wc = len(content.split())
    print(f"  {path} -> {wc} words")

def wa(lang, part, content):
    """Append to file."""
    path = os.path.join(BASE, lang, 'chapters', f'chapter-{part}.md')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content)

def write_appendix(lang, content):
    path = os.path.join(BASE, lang, 'appendices.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    wc = len(content.split())
    print(f"  {path} -> {wc} words")

print("Generating chapters...")
exec(open(os.path.join(BASE, '..', 'gen_ch20_en.py'), encoding='utf-8').read())
print("Done!")
