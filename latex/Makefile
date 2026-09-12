# Makefile for AI Architect Textbook

# Configuration
LATEX = pdflatex
BIBER = biber
LATEXFLAGS = -interaction=nonstopmode

# Main file
MAIN = main

# Default target
all: $(MAIN).pdf

# Build PDF
$(MAIN).pdf: $(MAIN).tex chapters/*.tex appendix.tex references.bib
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(BIBER) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.toc *.run.xml *.fdb_latexmk *.fls *.synctex.gz

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# Watch for changes (requires entr or similar)
watch:
	@echo "Watching for changes... (press Ctrl+C to stop)"
	@while true; do \
		inotifywait -qre modify .; \
		make; \
	done

# Quick build without bibliography
quick:
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build the PDF (default)"
	@echo "  clean    - Remove auxiliary files"
	@echo "  distclean - Remove everything including PDF"
	@echo "  quick    - Quick build without bibliography"
	@echo "  watch    - Watch for changes and rebuild"
	@echo "  help     - Show this help"

.PHONY: all clean distclean quick watch help