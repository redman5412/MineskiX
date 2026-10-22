"""Red Bull Wings Cup — player photo pipeline.

  projects/redbull/redbullphotosraw/      raw photos as received (any size/shape, jpg/png/webp)
  projects/redbull/redbullphotos_editted/ output: 1000x1000 transparent PNG, same base name

For each raw photo: remove the background (rembg, BiRefNet-portrait model), find the face (OpenCV YuNet),
then scale + place the cut-out on the broadcast photo template:
  top of head at 4% (40 px), chin at 40% (400 px), face centred, shoulders running off the bottom edge
(the name plate sits there). This is the framing every overlay photo box expects (see CLAUDE.md, Red Bull
section), so processed photos need no crop in the Roster tab.

Run (tools live outside the repo in C:/Users/mineski-prod/.redbull-photo-tool — venv + models):
  C:/Users/mineski-prod/.redbull-photo-tool/venv/Scripts/python.exe projects/redbull/tools/process_player_photos.py
Options: --force (re-process files that already have an output), --only NAME (one file), --debug (also
write a *_debug.jpg showing the detected face/head-top/chin over the original).
Already-processed photos are skipped, so just re-run it whenever new raws arrive.
"""
import argparse
import os
import sys

TOOL_HOME = r'C:\Users\mineski-prod\.redbull-photo-tool'
os.environ.setdefault('U2NET_HOME', TOOL_HOME)   # rembg looks for <U2NET_HOME>/models/<name>/<name>.onnx

import numpy as np                                  # noqa: E402
import cv2                                          # noqa: E402
from PIL import Image, ImageOps                     # noqa: E402
from rembg import new_session, remove               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RAW = os.path.join(PROJECT, 'redbullphotosraw')
OUT = os.path.join(PROJECT, 'redbullphotos_editted')
YUNET = os.path.join(TOOL_HOME, 'models', 'face_detection_yunet_2023mar.onnx')

SIZE = 1000          # output canvas (square)
HEAD_TOP = 0.04      # top of head, fraction of canvas height
CHIN = 0.40          # chin, fraction of canvas height
EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def detect_face(rgb):
    """Largest face -> (x, y, w, h) in pixels, or None. YuNet's box runs roughly brow-to-chin."""
    h, w = rgb.shape[:2]
    det = cv2.FaceDetectorYN.create(YUNET, '', (w, h), score_threshold=0.6)
    _, faces = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if faces is None or not len(faces):
        return None
    f = max(faces, key=lambda r: r[2] * r[3])
    return tuple(float(v) for v in f[:4])


def head_top(alpha, face):
    """Topmost cut-out row above the face, within the face's column span (hair / caps included)."""
    x, y, w, _ = face
    cols = alpha[:, max(0, int(x)):int(x + w)]
    rows = np.where((cols > 128).any(axis=1))[0]
    rows = rows[rows <= y]
    return float(rows[0]) if len(rows) else y - 0.35 * w   # no hair found above the box: estimate


def neck_fallback(alpha):
    """No face found: head top = first opaque row; chin ~ narrowest row of the upper figure (the neck)."""
    rows = np.where((alpha > 128).any(axis=1))[0]
    top, bottom = rows[0], rows[-1]
    widths = (alpha[top:bottom] > 128).sum(axis=1)
    span = max(10, int((bottom - top) * 0.45))
    seg = widths[int(span * 0.15):span]
    neck = top + int(span * 0.15) + int(np.argmin(seg))
    cols = np.where(alpha[neck] > 128)[0]
    cx = float(cols.mean()) if len(cols) else alpha.shape[1] / 2
    return float(top), float(neck) - 0.03 * (neck - top), cx


def process(src, dst, session, debug=False):
    im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    rgb = np.array(im)
    cut = remove(im, session=session, post_process_mask=True)           # RGBA, background removed
    alpha = np.array(cut.getchannel('A'))
    notes = []

    face = detect_face(rgb)
    if face:
        top = head_top(alpha, face)
        chin = face[1] + face[3]
        cx = face[0] + face[2] / 2
    else:
        top, chin, cx = neck_fallback(alpha)
        notes.append('no face detected - positioned from the body outline, check it')

    s = (CHIN - HEAD_TOP) * SIZE / max(1.0, chin - top)                  # source px -> canvas px
    new_w, new_h = max(1, round(cut.width * s)), max(1, round(cut.height * s))
    scaled = cut.resize((new_w, new_h), Image.LANCZOS)
    ox = round(SIZE / 2 - cx * s)
    oy = round(HEAD_TOP * SIZE - top * s)
    canvas = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    # alpha_composite (not paste-with-mask, which squares soft edge alpha); source offset handles overhang
    canvas.alpha_composite(scaled, dest=(max(ox, 0), max(oy, 0)), source=(max(-ox, 0), max(-oy, 0)))
    canvas.save(dst, optimize=True)

    # Quality checks: things that would look wrong in the overlay photo boxes.
    a = np.array(canvas.getchannel('A'))
    if s > 1.6:
        notes.append(f'low-res source - upscaled {s:.1f}x, may look soft (ask for a bigger photo)')
    if not (a[-1] > 128).any():
        notes.append('figure ends above the bottom edge - shoulders will not reach the name plate')
    for side, src_col in (('left', alpha[:, 0]), ('right', alpha[:, -1])):
        rows = np.where(src_col > 128)[0]
        if len(rows) and ((rows * s + oy) < SIZE).any() and (rows * s + oy > 0).any():
            x_on = ox if side == 'left' else ox + new_w
            if 0 < x_on < SIZE:
                notes.append(f'body is cut off at the {side} edge of the original photo (straight edge visible)')

    if debug:
        dbg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if face:
            x, y, w, h = map(int, face)
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.line(dbg, (0, int(top)), (dbg.shape[1], int(top)), (255, 230, 0), 2)
        cv2.line(dbg, (0, int(chin)), (dbg.shape[1], int(chin)), (255, 0, 200), 2)
        cv2.imwrite(os.path.splitext(dst)[0] + '_debug.jpg', dbg)
    return s, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW) if f.lower().endswith(EXTS) and (not args.only or f == args.only))
    if not files:
        print('No raw photos in', RAW)
        return
    session = new_session('birefnet-portrait')
    for f in files:
        dst = os.path.join(OUT, os.path.splitext(f)[0] + '.png')
        if os.path.exists(dst) and not args.force:
            print(f'skip  {f} (already done; --force to redo)')
            continue
        try:
            s, notes = process(os.path.join(RAW, f), dst, session, args.debug)
            print(f'done  {f} -> {os.path.basename(dst)}  (scale {s:.2f}x)')
            for n in notes:
                print('      ! ' + n)
        except Exception as e:  # keep going with the rest of the batch
            print(f'FAIL  {f}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
