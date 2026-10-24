"""Collapse overlapping predictions for the same card without mixing cards."""


def document_geometry(document):
    import cv2
    import numpy as np

    polygon = getattr(getattr(document, "polygon", None), "coordinates", None)
    if polygon is None:
        return None, 0.0
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        return None, 0.0
    hull = cv2.convexHull(points)
    area = float(cv2.contourArea(hull))
    return (hull, area) if area > 0 else (None, 0.0)


def nested_regions(first, second):
    import cv2

    first_polygon, first_area = document_geometry(first)
    second_polygon, second_area = document_geometry(second)
    if first_polygon is None or second_polygon is None:
        return False
    intersection, _ = cv2.intersectConvexConvex(first_polygon, second_polygon)
    return intersection / min(first_area, second_area) >= 0.9


def deduplicate_documents(documents, iou_threshold=0.65):
    import cv2

    candidates = [(document, *document_geometry(document)) for document in documents]
    # Prefer the full card over a slightly smaller duplicate crop, so printed
    # fields near the edge remain available. Distinct regions stay distinct.
    candidates.sort(key=lambda item: item[2], reverse=True)
    unique = []
    for document, polygon, area in candidates:
        duplicate = False
        if polygon is not None:
            for _previous, previous_polygon, previous_area in unique:
                if previous_polygon is None:
                    continue
                intersection, _ = cv2.intersectConvexConvex(polygon, previous_polygon)
                union = area + previous_area - intersection
                if union > 0 and intersection / union >= iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            unique.append((document, polygon, area))
    return [document for document, _polygon, _area in unique]
