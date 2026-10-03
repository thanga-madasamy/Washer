"""Offline validation: run AUTO mode on every photo in ref_images/ and captures/.

Every flange hole in each photo is located automatically and inspected; an annotated
copy (circle colour = result, number = annulus/reference ratio) is written to results/.
"""
import collections
import glob
import os

import cv2

import washer_detector as wd
from washer_poc import COLORS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(HERE, "ref_images", "*.jpeg"))) + \
        sorted(glob.glob(os.path.join(HERE, "captures", "*_raw.png")))
    for path in paths:
        img = cv2.imread(path)
        results = wd.inspect_auto(wd.to_gray(img))
        for res in results:
            color = COLORS[res.status]
            c = tuple(int(v) for v in res.center)
            cv2.circle(img, c, int(1.6 * res.hole_radius), color, 3)
            cv2.putText(img, f"{res.ratio:.2f}", (c[0] - 20, c[1] - int(2 * res.hole_radius)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{name}_result.jpg"), img)
        counts = dict(collections.Counter(r.status for r in results))
        print(f"{name:45s} {counts}")


if __name__ == "__main__":
    main()
