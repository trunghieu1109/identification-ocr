"""Order text boxes by their baselines, independently of absolute Y buckets."""
import math
from statistics import median


def _points(line):
    try:
        points = [(float(x), float(y)) for x, y in line.get("box", [])]
        if len(points) == 4 and all(math.isfinite(n) for point in points for n in point):
            if max(x for x, _ in points) > min(x for x, _ in points) and max(y for _, y in points) > min(y for _, y in points):
                return points
    except (TypeError, ValueError):
        pass
    return None


def projected_bounds(lines):
    """Deskew coordinates by the median text slope without resampling pixels."""
    points = [_points(line) for line in lines]
    if not all(points):
        return None
    slopes = []
    for box in points:
        dx = (box[1][0] + box[2][0] - box[0][0] - box[3][0]) / 2
        dy = (box[1][1] + box[2][1] - box[0][1] - box[3][1]) / 2
        height = max(y for _, y in box) - min(y for _, y in box)
        if dx >= 2 * height and abs(dy / dx) <= .5:
            slopes.append(dy / dx)
    slope = median(slopes) if slopes else 0
    bounds = []
    for box in points:
        ys = [y - slope * x for x, y in box]
        bounds.append((min(x for x, _ in box), min(ys), max(x for x, _ in box), max(ys)))
    return bounds


def visual_rows(lines):
    """Group overlapping baselines, then read each row from left to right.

    Project onto the median text slope so a tilted row still keeps its last
    character. Median row bounds prevent a tall box from bridging two rows.
    Inputs without usable boxes retain their original sequence.
    """
    bounds = projected_bounds(lines)
    if bounds is None:
        return [[line] for line in lines]
    positioned = [(box[1], box[3], box[0], line) for line, box in zip(lines, bounds)]
    rows = []
    for item in sorted(positioned, key=lambda item: ((item[0] + item[1]) / 2, item[2])):
        top, bottom, _left, _line = item
        matches = []
        for index, row in enumerate(rows):
            row_top, row_bottom = median(p[0] for p in row), median(p[1] for p in row)
            overlap = min(bottom, row_bottom) - max(top, row_top)
            distance = abs((top + bottom - row_top - row_bottom) / 2)
            if (overlap >= .45 * min(bottom - top, row_bottom - row_top)
                    and distance <= .6 * max(bottom - top, row_bottom - row_top)):
                matches.append((distance, index))
        if matches:
            rows[min(matches)[1]].append(item)
        else:
            rows.append([item])
    rows.sort(key=lambda row: median((item[0] + item[1]) / 2 for item in row))
    return [[item[3] for item in sorted(row, key=lambda item: item[2])] for row in rows]


def reading_order(lines):
    return [line for row in visual_rows(lines) for line in row]
