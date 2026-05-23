# Build the governing-equations PDF from LaTeX source.
# Prefers tectonic (single self-contained binary, no system TeX install needed);
# falls back to latexmk, then a two-pass pdflatex (for the table of contents).

TEX := governing-equations.tex
PDF := governing-equations.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX)
	@if command -v tectonic >/dev/null 2>&1; then \
	    tectonic $(TEX); \
	elif command -v latexmk >/dev/null 2>&1; then \
	    latexmk -pdf -interaction=nonstopmode $(TEX); \
	elif command -v pdflatex >/dev/null 2>&1; then \
	    pdflatex -interaction=nonstopmode $(TEX) && pdflatex -interaction=nonstopmode $(TEX); \
	else \
	    echo "No LaTeX toolchain found."; \
	    echo "Install tectonic (https://tectonic-typesetting.github.io/) -- a single"; \
	    echo "binary, no root needed -- or a TeX distribution (texlive)."; \
	    exit 1; \
	fi

clean:
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk
