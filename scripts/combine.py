import os
import glob

base = r"C:\Users\SCSM11\Desktop\AI-Architect-Textbook"
sections_dir = os.path.join(base, "sections", "en")
output_dir = os.path.join(base, "src", "en", "chapters")
os.makedirs(output_dir, exist_ok=True)

for ch_num in range(20, 26):
    pattern = os.path.join(sections_dir, f"ch{ch_num}_s*.md")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No sections found for chapter {ch_num}")
        continue
    parts = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            parts.append(fh.read())
    content = "\n\n".join(parts)
    out_path = os.path.join(output_dir, f"chapter-{ch_num}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    word_count = len(content.split())
    print(f"Chapter {ch_num}: {word_count} words -> {out_path}")
