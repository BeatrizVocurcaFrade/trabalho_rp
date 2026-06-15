.PHONY: run report slides pdf clean-latex

PYTHON ?= python3
PDFLATEX ?= pdflatex

run:
	$(PYTHON) -m src.malha_viaria --input data/raw --results data/results --graphs data/graphs --max-side 1400

report:
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=relatorio relatorio/main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=relatorio relatorio/main.tex

slides:
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=apresentacao apresentacao/main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=apresentacao apresentacao/main.tex

pdf: report slides

clean-latex:
	rm -f relatorio/*.aux relatorio/*.log relatorio/*.out relatorio/*.toc relatorio/*.nav relatorio/*.snm relatorio/*.vrb
	rm -f apresentacao/*.aux apresentacao/*.log apresentacao/*.out apresentacao/*.toc apresentacao/*.nav apresentacao/*.snm apresentacao/*.vrb
