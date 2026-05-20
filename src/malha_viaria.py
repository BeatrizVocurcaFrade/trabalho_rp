"""Pipeline classico para deteccao e reconstrucao de malha viaria.

O modulo foi escrito para ser usado pelo notebook do projeto, mas tambem
pode ser importado em scripts. A abordagem combina segmentacao por cor,
contraste local, textura, morfologia matematica e extracao de grafo a partir
do esqueleto da mascara de vias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class RoadDetectionResult:
    """Resultado consolidado do processamento de uma imagem."""

    image_name: str
    rgb: np.ndarray
    road_mask: np.ndarray
    class_map: np.ndarray
    skeleton: np.ndarray
    graph: nx.Graph
    segmented: np.ndarray
    overlay: np.ndarray
    graph_overlay: np.ndarray
    metrics: dict


def discover_project_root(start: Path | None = None) -> Path:
    """Encontra a raiz do projeto a partir do diretorio atual."""

    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "notebook").exists():
            return candidate
    return current


def list_images(raw_dir: Path) -> list[Path]:
    """Lista imagens de entrada sem depender de nomes fixos."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb_image(path: Path) -> np.ndarray:
    """Carrega imagem em RGB."""

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Nao foi possivel carregar a imagem: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_synthetic_demo(size: int = 768) -> np.ndarray:
    """Cria uma imagem sintetica simples quando nao ha imagens em data/raw.

    A demo serve apenas para validar a execucao do pipeline. Resultados reais
    devem ser obtidos com imagens de satelite adicionadas pelo usuario.
    """

    rng = np.random.default_rng(7)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[..., 0] = rng.normal(82, 14, (size, size)).clip(0, 255)
    base[..., 1] = rng.normal(115, 18, (size, size)).clip(0, 255)
    base[..., 2] = rng.normal(82, 14, (size, size)).clip(0, 255)

    asphalt = (92, 92, 88)
    dirt = (154, 116, 72)
    roads = [
        ((60, 140), (710, 170), 28, asphalt),
        ((120, 90), (520, 690), 22, asphalt),
        ((60, 520), (720, 460), 24, dirt),
        ((420, 40), (460, 720), 18, dirt),
        ((120, 650), (660, 260), 16, asphalt),
    ]
    for p1, p2, width, color in roads:
        cv2.line(base, p1, p2, color, width, lineType=cv2.LINE_AA)

    # Pequenos blocos urbanos para simular confundidores.
    for _ in range(45):
        x, y = rng.integers(0, size - 60, 2)
        w, h = rng.integers(18, 54, 2)
        color = tuple(int(v) for v in rng.normal([135, 128, 120], [25, 18, 18]))
        cv2.rectangle(base, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)

    return base


def resize_if_needed(rgb: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    """Reduz imagens muito grandes para manter o notebook responsivo."""

    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return rgb.copy(), 1.0

    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _local_std(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray_f * gray_f, (kernel_size, kernel_size))
    variance = np.maximum(mean_sq - mean * mean, 0)
    return np.sqrt(variance)


def preprocess_image(rgb: np.ndarray, max_side: int = 1400) -> dict:
    """Pre-processa a imagem e calcula espacos de cor usados na segmentacao."""

    resized, scale = resize_if_needed(rgb, max_side=max_side)
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=45, sigmaSpace=45)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    h, w = gray.shape
    local_kernel = _odd_kernel(min(h, w) // 45, minimum=9)

    return {
        "rgb": resized,
        "enhanced": enhanced,
        "gray": gray,
        "hsv": hsv,
        "lab": lab_equalized,
        "local_std": _local_std(gray, local_kernel),
        "scale": scale,
        "local_kernel": local_kernel,
    }


def segment_roads(pre: dict) -> np.ndarray:
    """Segmenta candidatos a via usando criterios classicos combinados."""

    rgb = pre["rgb"]
    gray = pre["gray"]
    hsv = pre["hsv"]
    local_std = pre["local_std"]
    h_channel, s_channel, v_channel = cv2.split(hsv)
    height, width = gray.shape
    min_side = min(height, width)

    # Regioes de vias tendem a ser alongadas e relativamente homogeneas.
    texture_limit = np.percentile(local_std, 55)
    homogeneous = local_std <= max(12, texture_limit)
    low_saturation = s_channel <= max(42, np.percentile(s_channel, 45))
    visible = v_channel >= 35

    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    channel_spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    grayish = channel_spread <= 38

    # Cores amarronzadas cobrem parte das vias de terra.
    dirt_hue = ((h_channel <= 28) | (h_channel >= 165)) & (s_channel >= 25)
    dirt_rgb = (r > g - 8) & (g >= b - 12) & ((r - b) >= 18)
    dirt_like = dirt_hue & dirt_rgb & homogeneous & visible

    asphalt_like = low_saturation & grayish & homogeneous & visible

    # Evita vegetacao e corpos muito escuros, que sao confundidores comuns.
    excess_green = 2 * g - r - b
    vegetation = (excess_green > np.percentile(excess_green, 74)) & (g > r + 8)
    very_dark = v_channel < 25

    # Bordas ajudam a preservar continuidade ao redor de faixas lineares.
    canny_low = int(max(30, np.percentile(gray, 35) * 0.66))
    canny_high = int(max(canny_low + 30, np.percentile(gray, 85)))
    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd_kernel(min_side // 180), _odd_kernel(min_side // 180))
    )
    edge_support = cv2.dilate(edges, edge_kernel, iterations=1) > 0

    adaptive_block = _odd_kernel(min_side // 18, minimum=31)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block,
        -3,
    )
    adaptive_regions = adaptive > 0

    candidate = (asphalt_like | dirt_like | (homogeneous & edge_support & adaptive_regions))
    candidate &= ~vegetation
    candidate &= ~very_dark

    return postprocess_mask(candidate)


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Limpa ruido e fecha pequenas falhas da mascara binaria."""

    mask_uint8 = (mask.astype(np.uint8) * 255)
    height, width = mask.shape
    min_side = min(height, width)
    close_size = _odd_kernel(min_side // 120, minimum=5)
    open_size = _odd_kernel(min_side // 300, minimum=3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)

    min_area = max(80, int(height * width * 0.00025))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    keep = np.zeros_like(cleaned, dtype=bool)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        elongation = max(w, h) / max(1, min(w, h))
        if area >= min_area or (area >= min_area * 0.45 and elongation >= 3.0):
            keep |= labels == label

    return keep


def classify_road_type(rgb: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """Classifica pixels de via como asfalto ou terra por heuristicas simples.

    Retorno:
        0 = fundo, 1 = asfalto, 2 = terra.
    """

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    dirt = road_mask & (((h <= 30) | (h >= 165)) & (s >= 30)) & (r >= g - 10) & (g >= b - 18)
    asphalt = road_mask & ((s < 60) | (spread < 35)) & ~dirt

    labels = np.zeros(road_mask.shape, dtype=np.uint8)
    labels[asphalt] = 1
    labels[dirt] = 2

    remaining = road_mask & (labels == 0)
    labels[remaining & (s < 75)] = 1
    labels[remaining & (s >= 75)] = 2
    return labels


def make_segmented_image(class_map: np.ndarray) -> np.ndarray:
    """Converte mapa de classes em imagem colorida."""

    segmented = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    segmented[class_map == 1] = (40, 160, 255)  # asfalto em azul
    segmented[class_map == 2] = (220, 145, 40)  # terra em laranja
    return segmented


def make_overlay(rgb: np.ndarray, class_map: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    """Sobrepoe as classes detectadas na imagem original."""

    segmented = make_segmented_image(class_map)
    overlay = rgb.copy()
    road_pixels = class_map > 0
    overlay[road_pixels] = (
        (1 - alpha) * overlay[road_pixels].astype(np.float32)
        + alpha * segmented[road_pixels].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def compute_skeleton(road_mask: np.ndarray) -> np.ndarray:
    """Calcula o esqueleto binario da mascara de vias."""

    return skeletonize(road_mask > 0)


def _neighbors_8(point: tuple[int, int], shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    y, x = point
    height, width = shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < height and 0 <= xx < width:
                yield yy, xx


def _pixel_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    counts = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return counts - skeleton.astype(np.uint8)


def skeleton_to_graph(skeleton: np.ndarray, min_edge_length: int = 4) -> nx.Graph:
    """Extrai um grafo a partir do esqueleto da malha viaria."""

    skeleton = skeleton.astype(bool)
    degree = _skeleton_degree(skeleton)
    node_mask = skeleton & ((degree == 1) | (degree >= 3))
    num_labels, labels = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)

    graph = nx.Graph()
    if num_labels <= 1:
        return graph

    node_label_to_id: dict[int, str] = {}
    for component_label in range(1, num_labels):
        ys, xs = np.where(labels == component_label)
        if len(xs) == 0:
            continue
        node_id = str(component_label - 1)
        component_degrees = degree[ys, xs]
        node_type = "intersecao" if int(component_degrees.max()) >= 3 else "extremidade"
        node_label_to_id[component_label] = node_id
        graph.add_node(
            node_id,
            x=float(xs.mean()),
            y=float(ys.mean()),
            tipo=node_type,
            pixels=int(len(xs)),
        )

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    shape = skeleton.shape

    for component_label, source_id in node_label_to_id.items():
        source_pixels = list(zip(*np.where(labels == component_label)))
        for source_pixel in source_pixels:
            for neighbor in _neighbors_8(source_pixel, shape):
                if not skeleton[neighbor]:
                    continue
                if labels[neighbor] == component_label:
                    continue
                first_edge = _pixel_edge(source_pixel, neighbor)
                if first_edge in visited_edges:
                    continue

                path = [source_pixel, neighbor]
                previous = source_pixel
                current = neighbor
                visited_edges.add(first_edge)

                while True:
                    current_label = int(labels[current])
                    if current_label > 0 and current_label != component_label:
                        target_id = node_label_to_id.get(current_label)
                        if target_id is not None and target_id != source_id and len(path) >= min_edge_length:
                            points = [(int(x), int(y)) for y, x in path]
                            length = _path_length_px(points)
                            _add_or_update_edge(graph, source_id, target_id, points, length)
                        break

                    next_pixels = [
                        pixel
                        for pixel in _neighbors_8(current, shape)
                        if skeleton[pixel] and pixel != previous
                    ]
                    if not next_pixels:
                        break

                    # Em regioes ambiguas, escolhe o proximo pixel ainda nao visitado.
                    next_pixel = None
                    for candidate in next_pixels:
                        if _pixel_edge(current, candidate) not in visited_edges:
                            next_pixel = candidate
                            break
                    if next_pixel is None:
                        break

                    visited_edges.add(_pixel_edge(current, next_pixel))
                    previous, current = current, next_pixel
                    path.append(current)

    return graph


def _path_length_px(points: list[tuple[int, int]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        total += math.hypot(x2 - x1, y2 - y1)
    return float(total)


def _add_or_update_edge(
    graph: nx.Graph,
    source_id: str,
    target_id: str,
    points: list[tuple[int, int]],
    length: float,
) -> None:
    geometry = " ".join(f"{x},{y}" for x, y in points)
    if graph.has_edge(source_id, target_id):
        old_length = float(graph[source_id][target_id].get("length_px", 0))
        if length <= old_length:
            graph[source_id][target_id].update(
                length_px=length,
                points_count=len(points),
                geometry=geometry,
            )
    else:
        graph.add_edge(
            source_id,
            target_id,
            length_px=length,
            points_count=len(points),
            geometry=geometry,
        )


def largest_connected_subgraph(graph: nx.Graph) -> nx.Graph:
    """Mantem o maior componente conectado do grafo."""

    if graph.number_of_nodes() == 0:
        return graph.copy()
    if nx.is_connected(graph):
        return graph.copy()
    largest_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_nodes).copy()


def draw_graph_overlay(rgb: np.ndarray, graph: nx.Graph) -> np.ndarray:
    """Desenha vertices e arestas do grafo sobre a imagem."""

    canvas = rgb.copy()
    for _, _, data in graph.edges(data=True):
        geometry = data.get("geometry", "")
        points = _parse_geometry(geometry)
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(255, 40, 40), thickness=3)

    for _, data in graph.nodes(data=True):
        x = int(round(float(data.get("x", 0))))
        y = int(round(float(data.get("y", 0))))
        color = (255, 255, 30) if data.get("tipo") == "intersecao" else (60, 255, 255)
        cv2.circle(canvas, (x, y), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return canvas


def _parse_geometry(geometry: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for pair in geometry.split():
        try:
            x_str, y_str = pair.split(",")
            points.append((int(x_str), int(y_str)))
        except ValueError:
            continue
    return points


def graph_to_json_dict(graph: nx.Graph, image_name: str, metrics: dict | None = None) -> dict:
    """Serializa o grafo em estrutura simples para JSON."""

    nodes = [
        {
            "id": node_id,
            "x": float(data.get("x", 0)),
            "y": float(data.get("y", 0)),
            "tipo": data.get("tipo", "desconhecido"),
            "pixels": int(data.get("pixels", 0)),
        }
        for node_id, data in graph.nodes(data=True)
    ]
    edges = []
    for source, target, data in graph.edges(data=True):
        points = [[x, y] for x, y in _parse_geometry(data.get("geometry", ""))]
        edges.append(
            {
                "source": source,
                "target": target,
                "length_px": float(data.get("length_px", 0)),
                "points": points,
            }
        )

    return {
        "image": image_name,
        "metrics": metrics or {},
        "nodes": nodes,
        "edges": edges,
    }


def save_outputs(result: RoadDetectionResult, results_dir: Path, graphs_dir: Path) -> None:
    """Salva imagens e grafo em disco."""

    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_name).stem

    cv2.imwrite(str(results_dir / f"{stem}_segmentada.png"), cv2.cvtColor(result.segmented, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(results_dir / f"{stem}_sobreposicao.png"), cv2.cvtColor(result.overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(results_dir / f"{stem}_grafo.png"), cv2.cvtColor(result.graph_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(results_dir / f"{stem}_mascara.png"), (result.road_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(results_dir / f"{stem}_esqueleto.png"), (result.skeleton.astype(np.uint8) * 255))

    with (graphs_dir / f"{stem}_grafo.json").open("w", encoding="utf-8") as file:
        json.dump(graph_to_json_dict(result.graph, result.image_name, result.metrics), file, ensure_ascii=False, indent=2)

    nx.write_graphml(result.graph, graphs_dir / f"{stem}_grafo.graphml")


def process_rgb_image(image_name: str, rgb: np.ndarray, max_side: int = 1400) -> RoadDetectionResult:
    """Executa o pipeline completo em uma imagem RGB."""

    pre = preprocess_image(rgb, max_side=max_side)
    road_mask = segment_roads(pre)
    class_map = classify_road_type(pre["rgb"], road_mask)
    skeleton = compute_skeleton(road_mask)
    full_graph = skeleton_to_graph(skeleton)
    graph = largest_connected_subgraph(full_graph)

    segmented = make_segmented_image(class_map)
    overlay = make_overlay(pre["rgb"], class_map)
    graph_overlay = draw_graph_overlay(overlay, graph)

    road_pixels = int(road_mask.sum())
    image_pixels = int(road_mask.size)
    metrics = {
        "scale": float(pre["scale"]),
        "road_pixel_ratio": float(road_pixels / max(1, image_pixels)),
        "road_pixels": road_pixels,
        "nodes_total": int(graph.number_of_nodes()),
        "edges_total": int(graph.number_of_edges()),
        "components_before_largest": int(nx.number_connected_components(full_graph)) if full_graph.number_of_nodes() else 0,
        "asphalt_pixels": int((class_map == 1).sum()),
        "dirt_pixels": int((class_map == 2).sum()),
    }

    return RoadDetectionResult(
        image_name=image_name,
        rgb=pre["rgb"],
        road_mask=road_mask,
        class_map=class_map,
        skeleton=skeleton,
        graph=graph,
        segmented=segmented,
        overlay=overlay,
        graph_overlay=graph_overlay,
        metrics=metrics,
    )


def process_image_file(path: Path, max_side: int = 1400) -> RoadDetectionResult:
    """Executa o pipeline completo em um arquivo de imagem."""

    rgb = load_rgb_image(path)
    return process_rgb_image(path.name, rgb, max_side=max_side)


def process_dataset(
    raw_dir: Path,
    results_dir: Path,
    graphs_dir: Path,
    max_side: int = 1400,
    use_synthetic_if_empty: bool = True,
) -> tuple[list[RoadDetectionResult], pd.DataFrame]:
    """Processa todas as imagens da pasta de entrada."""

    image_paths = list_images(raw_dir)
    results: list[RoadDetectionResult] = []

    if not image_paths and use_synthetic_if_empty:
        synthetic = build_synthetic_demo()
        result = process_rgb_image("demo_sintetico.png", synthetic, max_side=max_side)
        save_outputs(result, results_dir, graphs_dir)
        results.append(result)
    else:
        for image_path in image_paths:
            result = process_image_file(image_path, max_side=max_side)
            save_outputs(result, results_dir, graphs_dir)
            results.append(result)

    summary = pd.DataFrame(
        [{"image": result.image_name, **result.metrics} for result in results]
    )
    return results, summary


def plot_result(result: RoadDetectionResult, figsize: tuple[int, int] = (16, 10)) -> None:
    """Mostra as principais saidas visuais para uma imagem."""

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    views = [
        ("Imagem original", result.rgb),
        ("Mascara de vias", result.road_mask, "gray"),
        ("Classes: asfalto/terra", result.segmented),
        ("Sobreposicao", result.overlay),
        ("Esqueleto", result.skeleton, "gray"),
        ("Grafo extraido", result.graph_overlay),
    ]
    for axis, item in zip(axes, views):
        title = item[0]
        image = item[1]
        cmap = item[2] if len(item) > 2 else None
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(result.image_name)
    fig.tight_layout()


def plot_all_results(results: list[RoadDetectionResult]) -> None:
    """Mostra os resultados de todas as imagens processadas."""

    for result in results:
        plot_result(result)
