#!/usr/bin/env python3
"""
Highlight faint ink characters on a carbonized papyrus fragment photo.

Pipeline:
  1. Isolate the fragment (dark blob) from the white mount / clips.
  2. Boost local contrast (CLAHE) inside the fragment only.
  3. Extract thin, locally-different strokes (black-hat + top-hat, i.e. both
     darker-than-surroundings and lighter-than-surroundings strokes).
  4. Clean up noise, keep stroke-sized connected components.
  5. Save an overlay (colored strokes + boxes) and a contrast-enhanced image.

Usage:
    python highlight_characters.py artifact.jpg
    python highlight_characters.py artifact.jpg --kernel 15 --thresh 12 --min-area 25
    python highlight_characters.py artifact.jpg --debug      # saves intermediate masks

Requires: pip install opencv-python numpy
"""
import argparse
import os

import cv2
import numpy as np


def find_fragment_mask(gray, dark_thresh=None):
    """Return a binary mask of the dark fragment (largest dark blob)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if dark_thresh is None:
        # Otsu separates the dark charred papyrus from the bright mount
        dark_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_thresh = min(dark_thresh, 110)  # the fragment is very dark; stay conservative
    mask = (blur < dark_thresh).astype(np.uint8) * 255

    # Keep only the largest connected component (the fragment itself)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise RuntimeError("No dark fragment found; try --dark-thresh.")
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255

    # Fill holes (ink strokes can look like holes) and smooth the outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return filled


def extract_strokes(gray, frag_mask, kernel_size, thresh):
    """Detect thin local-contrast strokes inside the fragment."""
    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # Stretch the fragment's narrow intensity range first
    vals = gray[frag_mask > 0]
    lo, hi = np.percentile(vals, [1, 99])
    stretched = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    enhanced = clahe.apply(stretched)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, k)  # darker strokes
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, k)      # lighter strokes
    strokes = cv2.max(blackhat, tophat)

    # Erode the fragment mask so the fragment's edge is not detected as a "stroke"
    inner = cv2.erode(frag_mask, np.ones((15, 15), np.uint8))
    strokes = cv2.bitwise_and(strokes, strokes, mask=inner)

    _, binary = cv2.threshold(strokes, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return enhanced, strokes, binary


def filter_components(binary, min_area, max_area):
    """Keep connected components that are plausibly letter strokes."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        keep[labels == i] = 255
        boxes.append((x, y, w, h))
    return keep, boxes


def merge_boxes(boxes, shape, dilate=25):
    """Group nearby strokes into character-level boxes."""
    if not boxes:
        return []
    canvas = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)
    canvas = cv2.dilate(canvas, np.ones((dilate, dilate), np.uint8))
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged = []
    pad = dilate // 2
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        merged.append((x + pad, y + pad, max(w - 2 * pad, 1), max(h - 2 * pad, 1)))
    return merged


def main():
    ap = argparse.ArgumentParser(description="Highlight characters on a papyrus fragment photo.")
    ap.add_argument("image", help="Path to the artifact photo")
    ap.add_argument("-o", "--output", default=None, help="Output overlay path")
    ap.add_argument("--kernel", type=int, default=15, help="Stroke-scale kernel size in px (odd; ~stroke length)")
    ap.add_argument("--thresh", type=int, default=14, help="Stroke detection threshold (lower = more sensitive)")
    ap.add_argument("--min-area", type=int, default=30, help="Min stroke component area (px)")
    ap.add_argument("--max-area", type=int, default=4000, help="Max stroke component area (px)")
    ap.add_argument("--dark-thresh", type=int, default=None, help="Manual threshold for fragment segmentation")
    ap.add_argument("--color", default="0,255,255", help="Highlight color as B,G,R (default yellow)")
    ap.add_argument("--debug", action="store_true", help="Save intermediate images")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    color = tuple(int(c) for c in args.color.split(","))
    if args.kernel % 2 == 0:
        args.kernel += 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frag = find_fragment_mask(gray, args.dark_thresh)
    enhanced, strokes, binary = extract_strokes(gray, frag, args.kernel, args.thresh)
    ink, boxes = filter_components(binary, args.min_area, args.max_area)
    chars = merge_boxes(boxes, gray.shape)

    # Overlay: tint ink strokes and draw character boxes
    overlay = img.copy()
    overlay[ink > 0] = color
    result = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    for x, y, w, h in chars:
        cv2.rectangle(result, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 2)

    # Outline the fragment itself
    cnts, _ = cv2.findContours(frag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cnts, -1, (0, 255, 0), 1)

    base, _ = os.path.splitext(args.image)
    out = args.output or f"{base}_highlighted.png"
    cv2.imwrite(out, result)

    # Contrast-enhanced crop of the fragment (useful for reading the letters by eye)
    x, y, w, h = cv2.boundingRect(frag)
    crop = enhanced[y:y + h, x:x + w]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    enh_out = f"{base}_enhanced.png"
    cv2.imwrite(enh_out, crop)

    if args.debug:
        cv2.imwrite(f"{base}_dbg_fragment_mask.png", frag)
        cv2.imwrite(f"{base}_dbg_strokes.png", strokes)
        cv2.imwrite(f"{base}_dbg_binary.png", binary)
        cv2.imwrite(f"{base}_dbg_ink.png", ink)

    print(f"Stroke components kept : {len(boxes)}")
    print(f"Character regions      : {len(chars)}")
    print(f"Saved overlay          : {out}")
    print(f"Saved enhanced crop    : {enh_out}")
    print("Tip: if too much/too little is highlighted, adjust --thresh and --kernel.")


if __name__ == "__main__":
    main()
