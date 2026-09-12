# AI Architect Textbook - LaTeX Source

This directory contains the LaTeX source files for the AI Architect Textbook.

## Prerequisites

- TeX Live or MiKTeX installed
- Biber (for bibliography management)
- PDF viewer (for viewing the output)

## Building the PDF

### Windows
```cmd
build.bat
```

### Linux/macOS
```bash
make
```

Or manually:
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Project Structure

```
latex/
├── main.tex           # Main LaTeX file
├── chapters/          # Chapter files (chapter-01.tex to chapter-25.tex)
├── appendix.tex       # Appendices
├── references.bib     # Bibliography database
├── figures/           # Images and figures
├── build.bat          # Windows build script
├── Makefile           # Unix build script
└── README.md          # This file
```

## Adding Content

1. Edit the chapter files in `chapters/` directory
2. Replace the TODO placeholders with actual content
3. Add figures to the `figures/` directory
4. Add references to `references.bib`
5. Rebuild the PDF

## Customization

- Edit `main.tex` to change document settings
- Modify chapter styling in the `titlesec` configuration
- Update bibliography style in `biblatex` options

## Troubleshooting

- If you get missing package errors, install the required packages
- If bibliography doesn't appear, run `biber main` manually
- Check the `.log` file for detailed error messages