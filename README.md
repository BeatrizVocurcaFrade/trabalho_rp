# Deteccao e Reconstrucao de Malha Viaria

Projeto pratico da disciplina de Reconhecimento de Padroes para detectar vias em imagens de satelite, gerar uma segmentacao visual e reconstruir uma representacao em grafo da malha viaria.

## Estrutura

- `notebook/deteccao_malha_viaria.ipynb`: pipeline principal documentado.
- `data/raw/`: coloque aqui as imagens de entrada (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).
- `data/results/`: imagens segmentadas, sobreposicoes e visualizacoes do grafo.
- `data/graphs/`: grafos exportados em JSON e GraphML.
- `relatorio/main.tex`: relatorio tecnico em LaTeX.

## Como executar

1. Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Coloque uma ou mais imagens de satelite em `data/raw/`.

4. Abra o notebook:

```bash
jupyter notebook notebook/deteccao_malha_viaria.ipynb
```

5. Execute todas as celulas. Os resultados serao salvos em `data/results/` e `data/graphs/`.

## Observacao metodologica

A solucao usa visao computacional classica, sem treinamento de rede neural. Isso deixa o comportamento mais explicavel para o relatorio e permite executar o trabalho com imagens novas sem depender de uma base anotada.
