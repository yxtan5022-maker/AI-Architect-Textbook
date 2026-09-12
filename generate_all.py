#!/usr/bin/env python3
"""Procedural chapter generator - builds content from templates."""
import os, textwrap

BASE = r'C:\Users\SCSM11\Desktop\AI-Architect-Textbook\src'

def wf(lang, num, content):
    p = os.path.join(BASE, lang, 'chapters', f'chapter-{num}.md')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  {lang}/ch{num}: {len(content.split())} words, {len(content)} chars")

def wa(lang, num, content):
    p = os.path.join(BASE, lang, 'chapters', f'chapter-{num}.md')
    with open(p, 'a', encoding='utf-8') as f:
        f.write(content)

def wappendix(lang, content):
    p = os.path.join(BASE, lang, 'appendices.md')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  {lang}/appendices: {len(content.split())} words")

# Build chapters from section files
SECTION_DIR = os.path.join(BASE, '..', 'sections')

print("Generating chapters from section files...")
for lang in ['en', 'zh']:
    sdir = os.path.join(SECTION_DIR, lang)
    for ch in range(20, 26):
        parts = []
        for sec in range(1, 20):
            sf = os.path.join(sdir, f'ch{ch}_s{sec}.md')
            if os.path.exists(sf):
                with open(sf, 'r', encoding='utf-8') as f:
                    parts.append(f.read())
        if parts:
            wf(lang, ch, '\n\n'.join(parts) + '\n')
        else:
            print(f"  {lang}/ch{ch}: SKIPPED (no section files)")

# Appendices
for lang in ['en', 'zh']:
    af = os.path.join(sdir if os.path.exists(sdir) else SECTION_DIR, f'appendices_{lang}.md')
    if not os.path.exists(af):
        af = os.path.join(SECTION_DIR, f'appendices_{lang}.md')
    if os.path.exists(af):
        with open(af, 'r', encoding='utf-8') as f:
            wappendix(lang, f.read())

print("Done!")
