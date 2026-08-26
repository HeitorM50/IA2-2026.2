# Build do artigo. Rodar de dentro de artigo-1/paper/.
MAIN = main

.PHONY: all watch clean distclean

all:
	latexmk -pdf -interaction=nonstopmode -file-line-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c

distclean:
	latexmk -C
