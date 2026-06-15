"""Road detection and road-network reconstruction from satellite images.

The pipeline intentionally uses classical computer-vision operations. This
keeps the project reproducible without a training dataset and makes each stage
easy to inspect in the technical report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class RoadDetectionResult:
    image_name: str
    rgb: np.ndarray
    road_mask: np.ndarray
    class_map: np.ndarray
    skeleton: np.ndarray
    graph: nx.Graph
    segmented: np.ndarray
    overlay: np.ndarray
    graph_overlay: np.ndarray
    metrics: dict[str, Any]


def list_images(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Nao foi possivel carregar a imagem: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_synthetic_demo(size: int = 768) -> np.ndarray:
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

    for _ in range(45):
        x, y = rng.integers(0, size - 60, 2)
        width, height = rng.integers(18, 54, 2)
        color = tuple(
            int(v)
            for v in rng.normal([135, 128, 120], [25, 18, 18]).clip(0, 255)
        )
        cv2.rectangle(
            base,
            (int(x), int(y)),
            (int(x + width), int(y + height)),
            color,
            -1,
        )
    return base


def resize_if_needed(rgb: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    if max_side <= 0:
        raise ValueError("max_side deve ser maior que zero.")

    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return rgb.copy(), 1.0

    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA), scale


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _local_std(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray_f * gray_f, (kernel_size, kernel_size))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0))


def preprocess_image(rgb: np.ndarray, max_side: int = 1400) -> dict[str, Any]:
    resized, scale = resize_if_needed(rgb, max_side=max_side)
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=45, sigmaSpace=45)

    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge([l_equalized, a_channel, b_channel]),
        cv2.COLOR_LAB2RGB,
    )

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    height, width = gray.shape
    local_kernel = _odd_kernel(min(height, width) // 45, minimum=9)
    return {
        "rgb": resized,
        "enhanced": enhanced,
        "gray": gray,
        "hsv": hsv,
        "lab": cv2.merge([l_equalized, a_channel, b_channel]),
        "local_std": _local_std(gray, local_kernel),
        "scale": scale,
        "local_kernel": local_kernel,
    }


def compute_line_support(gray: np.ndarray) -> np.ndarray:
    """Return a [0, 1] map with high values over line-like structures."""

    height, width = gray.shape
    min_side = min(height, width)
    line_len = max(11, min_side // 55)

    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blur, 20, 80)

    directional = np.zeros((height, width), dtype=np.float32)
    for angle_deg in range(0, 180, 30):
        angle = np.deg2rad(angle_deg)
        dx = int(round(np.cos(angle) * line_len))
        dy = int(round(np.sin(angle) * line_len))
        half = max(abs(dx), abs(dy), 1)
        kernel_size = half * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = half
        cv2.line(
            kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1,
            1,
        )
        directional += cv2.dilate(edges, kernel).astype(np.float32)

    ridge = np.zeros((height, width), dtype=np.float32)
    for sigma in (2.0, 4.5):
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
        ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)

        trace = (ixx + iyy) / 2
        discriminant = np.sqrt(((ixx - iyy) / 2) ** 2 + ixy**2)
        lambda_1 = trace + discriminant
        lambda_2 = trace - discriminant

        second_order_energy = np.sqrt(lambda_1**2 + lambda_2**2)
        larger_abs = np.maximum(np.abs(lambda_1), np.abs(lambda_2))
        smaller_abs = np.minimum(np.abs(lambda_1), np.abs(lambda_2))
        blobness = smaller_abs / (larger_abs + 1e-9)
        response = second_order_energy / (second_order_energy.max() + 1e-9)
        ridge = np.maximum(ridge, response * np.exp(-(blobness**2) / 0.5))

    directional_norm = directional / (directional.max() + 1e-9)
    ridge_norm = ridge / (ridge.max() + 1e-9)
    combined = np.maximum(directional_norm * 0.7, ridge_norm * 0.6)
    return np.clip(combined / (combined.max() + 1e-9), 0.0, 1.0).astype(np.float32)


def segment_roads(preprocessed: dict[str, Any]) -> np.ndarray:
    rgb = preprocessed["rgb"]
    gray = preprocessed["gray"]
    hsv = preprocessed["hsv"]
    local_std = preprocessed["local_std"]

    h_channel, s_channel, v_channel = cv2.split(hsv)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)

    texture_limit = np.percentile(local_std, 55)
    homogeneous = local_std <= max(12, texture_limit)
    low_saturation = s_channel <= max(42, np.percentile(s_channel, 45))
    visible = v_channel >= 35

    spread = (
        np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    ).astype(np.int16)
    grayish = spread <= 38

    dirt_hue = ((h_channel <= 28) | (h_channel >= 165)) & (s_channel >= 20)
    dirt_rgb = (r > g - 10) & (g >= b - 14) & ((r - b) >= 15)
    dirt_like = dirt_hue & dirt_rgb & homogeneous & visible

    asphalt_like = low_saturation & grayish & homogeneous & visible

    excess_green = (2 * g - r - b).astype(np.int16)
    vegetation = (excess_green > np.percentile(excess_green, 70)) & (g > r + 6)
    shadow = (v_channel < 38) & (s_channel < 50)
    line_mask = compute_line_support(gray) > 0.15

    candidate = asphalt_like | (dirt_like & line_mask)
    candidate &= ~vegetation
    candidate &= ~shadow
    return _postprocess_mask(candidate)


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    height, width = mask.shape
    min_side = min(height, width)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(min_side // 120, 5),) * 2,
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(min_side // 300, 3),) * 2,
    )
    clean = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, open_kernel, iterations=1)

    total_pixels = height * width
    min_area = max(80, int(total_pixels * 0.00025))
    max_blob = int(total_pixels * 0.05)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8,
    )
    keep = np.zeros(mask.shape, dtype=bool)

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        bbox_width = stats[label, cv2.CC_STAT_WIDTH]
        bbox_height = stats[label, cv2.CC_STAT_HEIGHT]
        elongation = max(bbox_width, bbox_height) / max(1, min(bbox_width, bbox_height))
        fill = area / max(1, bbox_width * bbox_height)

        if area < min_area:
            continue
        if area > max_blob and elongation < 2.5:
            continue
        if area > max_blob * 0.4 and fill > 0.65 and elongation < 2.5:
            continue
        keep |= labels == label

    return keep


def classify_road_type(rgb: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, _ = cv2.split(hsv)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    dirt = (
        road_mask
        & (((h_channel <= 30) | (h_channel >= 165)) & (s_channel >= 30))
        & (r >= g - 10)
        & (g >= b - 18)
    )
    asphalt = road_mask & ((s_channel < 60) | (spread < 35)) & ~dirt

    labels = np.zeros(road_mask.shape, dtype=np.uint8)
    labels[asphalt] = 1
    labels[dirt] = 2
    remaining = road_mask & (labels == 0)
    labels[remaining & (s_channel < 75)] = 1
    labels[remaining & (s_channel >= 75)] = 2
    return labels


def make_segmented_image(class_map: np.ndarray) -> np.ndarray:
    segmented = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    segmented[class_map == 1] = (40, 160, 255)
    segmented[class_map == 2] = (220, 145, 40)
    return segmented


def make_overlay(rgb: np.ndarray, class_map: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    segmented = make_segmented_image(class_map)
    overlay = rgb.copy()
    road_pixels = class_map > 0
    overlay[road_pixels] = (
        (1 - alpha) * overlay[road_pixels].astype(np.float32)
        + alpha * segmented[road_pixels].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def compute_skeleton(road_mask: np.ndarray) -> np.ndarray:
    return skeletonize(road_mask > 0)


def _neighbors_8(point: tuple[int, int], shape: tuple[int, int]):
    y, x = point
    height, width = shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < height and 0 <= xx < width:
                yield yy, xx


def _pixel_edge(a: tuple[int, int], b: tuple[int, int]):
    return tuple(sorted((a, b)))


def _skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    counts = cv2.filter2D(
        skeleton.astype(np.uint8),
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return counts - skeleton.astype(np.uint8)


def skeleton_to_graph(skeleton: np.ndarray, min_edge_length: int = 4) -> nx.Graph:
    skeleton = skeleton.astype(bool)
    degree = _skeleton_degree(skeleton)
    node_mask = skeleton & ((degree == 1) | (degree >= 3))
    num_labels, labels = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)

    graph = nx.Graph()
    if num_labels <= 1:
        return graph

    node_label_to_id: dict[int, str] = {}
    for component in range(1, num_labels):
        ys, xs = np.where(labels == component)
        if len(xs) == 0:
            continue

        node_id = str(component - 1)
        node_type = "intersecao" if int(degree[ys, xs].max()) >= 3 else "extremidade"
        node_label_to_id[component] = node_id
        graph.add_node(
            node_id,
            x=float(xs.mean()),
            y=float(ys.mean()),
            tipo=node_type,
            pixels=int(len(xs)),
        )

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    shape = skeleton.shape
    for component, source_id in node_label_to_id.items():
        for source_pixel in zip(*np.where(labels == component)):
            for neighbor in _neighbors_8(source_pixel, shape):
                if not skeleton[neighbor] or labels[neighbor] == component:
                    continue

                first_edge = _pixel_edge(source_pixel, neighbor)
                if first_edge in visited:
                    continue

                path = [source_pixel, neighbor]
                previous, current = source_pixel, neighbor
                visited.add(first_edge)

                while True:
                    current_label = int(labels[current])
                    if current_label > 0 and current_label != component:
                        target_id = node_label_to_id.get(current_label)
                        if (
                            target_id
                            and target_id != source_id
                            and len(path) >= min_edge_length
                        ):
                            points = [(int(x), int(y)) for y, x in path]
                            length = sum(
                                math.hypot(x2 - x1, y2 - y1)
                                for (x1, y1), (x2, y2) in zip(points, points[1:])
                            )
                            geometry = " ".join(f"{x},{y}" for x, y in points)
                            if graph.has_edge(source_id, target_id):
                                current_length = float(
                                    graph[source_id][target_id].get(
                                        "length_px",
                                        float("inf"),
                                    )
                                )
                                if length < current_length:
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
                        break

                    next_pixels = [
                        pixel
                        for pixel in _neighbors_8(current, shape)
                        if skeleton[pixel] and pixel != previous
                    ]
                    if not next_pixels:
                        break

                    next_pixel = next(
                        (
                            pixel
                            for pixel in next_pixels
                            if _pixel_edge(current, pixel) not in visited
                        ),
                        None,
                    )
                    if next_pixel is None:
                        break

                    visited.add(_pixel_edge(current, next_pixel))
                    previous, current = current, next_pixel
                    path.append(current)

    return graph


def largest_connected_subgraph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0 or nx.is_connected(graph):
        return graph.copy()
    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


def _parse_geometry(geometry: str) -> list[tuple[int, int]]:
    points = []
    for pair in geometry.split():
        try:
            x_str, y_str = pair.split(",")
            points.append((int(x_str), int(y_str)))
        except ValueError:
            continue
    return points


def draw_graph_overlay(rgb: np.ndarray, graph: nx.Graph) -> np.ndarray:
    canvas = rgb.copy()
    for _, _, data in graph.edges(data=True):
        points = _parse_geometry(data.get("geometry", ""))
        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [np.array(points, dtype=np.int32).reshape((-1, 1, 2))],
                isClosed=False,
                color=(255, 40, 40),
                thickness=3,
            )

    for _, data in graph.nodes(data=True):
        x = int(round(float(data.get("x", 0))))
        y = int(round(float(data.get("y", 0))))
        color = (255, 255, 30) if data.get("tipo") == "intersecao" else (60, 255, 255)
        cv2.circle(canvas, (x, y), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return canvas


def save_outputs(
    result: RoadDetectionResult,
    results_dir: Path,
    graphs_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(result.image_name).stem
    cv2.imwrite(
        str(results_dir / f"{stem}_segmentada.png"),
        cv2.cvtColor(result.segmented, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_sobreposicao.png"),
        cv2.cvtColor(result.overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_grafo.png"),
        cv2.cvtColor(result.graph_overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_mascara.png"),
        result.road_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_esqueleto.png"),
        result.skeleton.astype(np.uint8) * 255,
    )

    nodes = [
        {
            "id": node_id,
            "x": float(data.get("x", 0)),
            "y": float(data.get("y", 0)),
            "tipo": data.get("tipo"),
            "pixels": int(data.get("pixels", 0)),
        }
        for node_id, data in result.graph.nodes(data=True)
    ]
    edges = [
        {
            "source": source,
            "target": target,
            "length_px": float(data.get("length_px", 0)),
            "points": [[x, y] for x, y in _parse_geometry(data.get("geometry", ""))],
        }
        for source, target, data in result.graph.edges(data=True)
    ]

    with (graphs_dir / f"{stem}_grafo.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "image": result.image_name,
                "metrics": result.metrics,
                "nodes": nodes,
                "edges": edges,
            },
            file,
            indent=2,
        )
    nx.write_graphml(result.graph, graphs_dir / f"{stem}_grafo.graphml")


def process_rgb_image(
    image_name: str,
    rgb: np.ndarray,
    max_side: int = 1400,
) -> RoadDetectionResult:
    preprocessed = preprocess_image(rgb, max_side=max_side)
    road_mask = segment_roads(preprocessed)
    class_map = classify_road_type(preprocessed["rgb"], road_mask)
    skeleton = compute_skeleton(road_mask)
    full_graph = skeleton_to_graph(skeleton)
    graph = largest_connected_subgraph(full_graph)
    segmented = make_segmented_image(class_map)
    overlay = make_overlay(preprocessed["rgb"], class_map)
    graph_overlay = draw_graph_overlay(overlay, graph)

    road_pixels = int(road_mask.sum())
    mask_components = (
        int(cv2.connectedComponents(road_mask.astype(np.uint8), connectivity=8)[0]) - 1
    )
    metrics = {
        "scale": float(preprocessed["scale"]),
        "road_pixel_ratio": float(road_pixels / max(1, road_mask.size)),
        "road_pixels": road_pixels,
        "mask_components": mask_components,
        "nodes_total": int(graph.number_of_nodes()),
        "edges_total": int(graph.number_of_edges()),
        "components_before_largest": (
            int(nx.number_connected_components(full_graph))
            if full_graph.number_of_nodes()
            else 0
        ),
        "asphalt_pixels": int((class_map == 1).sum()),
        "dirt_pixels": int((class_map == 2).sum()),
    }

    return RoadDetectionResult(
        image_name=image_name,
        rgb=preprocessed["rgb"],
        road_mask=road_mask,
        class_map=class_map,
        skeleton=skeleton,
        graph=graph,
        segmented=segmented,
        overlay=overlay,
        graph_overlay=graph_overlay,
        metrics=metrics,
    )


def process_dataset(
    raw_dir: Path,
    results_dir: Path,
    graphs_dir: Path,
    max_side: int = 1400,
    demo_if_empty: bool = True,
) -> tuple[list[RoadDetectionResult], pd.DataFrame]:
    image_paths = list_images(raw_dir)
    results: list[RoadDetectionResult] = []

    if not image_paths and demo_if_empty:
        result = process_rgb_image(
            "demo_sintetico.png",
            build_synthetic_demo(),
            max_side=max_side,
        )
        save_outputs(result, results_dir, graphs_dir)
        results.append(result)
    else:
        for image_path in image_paths:
            result = process_rgb_image(
                image_path.name,
                load_rgb_image(image_path),
                max_side=max_side,
            )
            save_outputs(result, results_dir, graphs_dir)
            results.append(result)

    summary = pd.DataFrame([{"image": result.image_name, **result.metrics} for result in results])
    return results, summary


def plot_all_results(results: list[RoadDetectionResult]) -> None:
    import matplotlib.pyplot as plt

    for result in results:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        views = [
            ("Imagem original", result.rgb, None),
            ("Mascara de vias", result.road_mask, "gray"),
            ("Classes: asfalto/terra", result.segmented, None),
            ("Sobreposicao", result.overlay, None),
            ("Esqueleto", result.skeleton, "gray"),
            ("Grafo extraido", result.graph_overlay, None),
        ]
        for axis, (title, image, cmap) in zip(axes.ravel(), views):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(result.image_name)
        fig.tight_layout()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detecta vias e reconstroi uma malha viaria em grafo.",
    )
    parser.add_argument("--input", default="data/raw", help="Pasta com imagens de entrada.")
    parser.add_argument("--results", default="data/results", help="Pasta para imagens geradas.")
    parser.add_argument("--graphs", default="data/graphs", help="Pasta para grafos JSON/GraphML.")
    parser.add_argument("--max-side", type=int, default=1400, help="Maior lado maximo da imagem.")
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Nao gera a imagem sintetica quando data/raw esta vazia.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _, summary = process_dataset(
        raw_dir=Path(args.input),
        results_dir=Path(args.results),
        graphs_dir=Path(args.graphs),
        max_side=args.max_side,
        demo_if_empty=not args.no_demo,
    )

    if summary.empty:
        print("Nenhuma imagem processada.")
    else:
        print(summary.to_string(index=False))
    print(f"Imagens geradas em: {Path(args.results).resolve()}")
    print(f"Grafos gerados em: {Path(args.graphs).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
