# Deteccao e Reconstrucao de Malha Viaria

Projeto pratico da disciplina de Reconhecimento de Padroes para detectar vias em imagens de satelite, gerar uma segmentacao visual e reconstruir uma representacao em grafo da malha viaria.

## Estrutura

- `src/malha_viaria.py`: pipeline executavel e reutilizavel.
- `notebook/deteccao_malha_viaria.ipynb`: roteiro didatico para executar e visualizar o pipeline.
- `data/raw/`: coloque aqui as imagens de entrada (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).
- `data/results/`: imagens segmentadas, sobreposicoes e visualizacoes do grafo.
- `data/graphs/`: grafos exportados em JSON e GraphML.
- `relatorio/main.tex`: relatorio tecnico em LaTeX.
- `slide/main.tex`: apresentacao Beamer em LaTeX.

## Como executar

1. Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Em Debian/Ubuntu, se `venv` ou `pip` nao estiverem instalados, instale antes os pacotes `python3-venv` e `python3-pip`.

2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Coloque uma ou mais imagens de satelite em `data/raw/`.

4. Execute o pipeline pela linha de comando:

```bash
python3 -m src.malha_viaria --input data/raw --results data/results --graphs data/graphs --max-side 1400
```

Tambem e possivel usar:

```bash
make run
```

Se `data/raw/` estiver vazia, o sistema gera uma imagem sintetica de demonstracao para validar o fluxo completo.

5. Opcionalmente, abra o notebook:

```bash
jupyter notebook notebook/deteccao_malha_viaria.ipynb
```

6. Execute todas as celulas. Os resultados serao salvos em `data/results/` e `data/graphs/`.

## Como gerar os PDFs

Para gerar somente o relatorio:

```bash
make report
```

Para gerar somente os slides:

```bash
make slides
```

Para gerar os dois PDFs:

```bash
make pdf
```

Os arquivos finais sao `relatorio/main.pdf` e `slide/main.pdf`. Eles sao artefatos locais e ficam ignorados pelo Git.

## Observacao metodologica

A solucao usa visao computacional classica, sem treinamento de rede neural. Isso deixa o comportamento mais explicavel para o relatorio e permite executar o trabalho com imagens novas sem depender de uma base anotada.
