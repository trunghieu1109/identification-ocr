"""Expanded single-line crops used by the v2-based DocU experiment."""
import math

STRATEGY = 'docu_expanded_v3'
PROCESSING = 'v2_expanded_box_v1'


def context_strips(box, image_size):
    """Read the whole row to the left, plus two separate caption rows above.

    Keep these as separate single-line images: feeding a tall block to a
    line recognizer can discard the very caption we are trying to restore.
    """
    width, height = image_size
    x1, y1, x2, y2 = box
    h = max(8, y2-y1)
    right = min(width, math.ceil(x2+max(4, .3*h)))
    bands = [(y1-.25*h, y2+.25*h),
             (y1-1.6*h, y1-.05*h),
             (y1-3.15*h, y1-1.6*h)]
    return [(0, max(0, math.floor(top)), right, min(height, math.ceil(bottom)))
            for top, bottom in bands if min(height, math.ceil(bottom))-max(0, math.floor(top)) > 3]

