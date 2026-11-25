"""Video -> ArcFace identity vectors, following the official TI2Net pipeline.

Per frame (same steps as TI2Net's utils/video2image_cropping.py + utils/libs.py):
  dlib frontal face detector (upsample 1) -> square box x1.3 around the face centre -> crop
  -> grayscale -> resize 128x128 -> [image, left-right flip] -> ArcFace ResNet-18 -> 512 + 512 = 1024-D.

Our choices (see DECISIONS.md): frames in temporal order; largest face when several are detected;
frames without a face are skipped; crops are not written to disk; one .npy per video (resumable).

Usage (FF++):
  python src/extract_ids.py --ffpp_root D:/FFpp --compression c23 --out_dir id_vectors/FFpp_c23
Usage (any folder of videos):
  python src/extract_ids.py --videos_dir D:/CelebDF/Celeb-real --name celeb_real --out_dir id_vectors/CelebDF
"""
import argparse
import csv
import os
import sys
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np

FFPP_SUBSETS = {
    "real": "original_sequences/youtube",
    "df": "manipulated_sequences/Deepfakes",
    "ff": "manipulated_sequences/Face2Face",
    "fs": "manipulated_sequences/FaceSwap",
    "nt": "manipulated_sequences/NeuralTextures",
}
SIZE = (128, 128)
SCALE = 1.3
_detector = None


def _init_worker():
    global _detector
    import dlib
    _detector = dlib.get_frontal_face_detector()


def crop_face(image, top, bottom, left, right):
    """Same box arithmetic as cropSingleFace in the official repo."""
    top, bottom, left, right = max(top, 0), max(bottom, 0), max(left, 0), max(right, 0)
    v_center, h_center = (top + bottom) // 2, (left + right) // 2
    size_bb = int(max(bottom - top, right - left) * SCALE)
    top = max(int(v_center - size_bb // 2), 0)
    bottom = min(int(v_center + size_bb // 2), image.shape[0])
    left = max(int(h_center - size_bb // 2), 0)
    right = min(int(h_center + size_bb // 2), image.shape[1])
    return image[top:bottom, left:right]


def detect(frame, detect_height, upsample):
    """Largest face as (top, bottom, left, right) in full-frame pixels, plus the number of faces.

    detect_height=0, upsample=1 is the authors' setting (detector on the full frame, upsampled once).
    detect_height>0 runs the detector on a copy shrunk to that height and scales the box back;
    if that finds nothing, the frame is retried with the authors' setting.
    """
    f = 1.0
    img = frame
    if detect_height and frame.shape[0] > detect_height:
        f = detect_height / frame.shape[0]
        img = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    faces = _detector(img, upsample)
    fallback = False
    if len(faces) == 0 and (f != 1.0 or upsample != 1):
        # nothing found with the fast setting: retry once with the authors' setting
        f, fallback = 1.0, True
        faces = _detector(frame, 1)
    if len(faces) == 0:
        return None, 0, fallback
    r = max(faces, key=lambda x: x.width() * x.height())
    box = tuple(int(round(v / f)) for v in (r.top(), r.bottom(), r.left(), r.right()))
    return box, len(faces), fallback


def process_video(job):
    """Returns (video_path, crops uint8 [n,128,128], stats dict). Runs in a worker process."""
    video_path, detect_height, upsample = job
    cap = cv2.VideoCapture(video_path)
    crops, frames, no_face, multi_face, fallbacks = [], 0, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        box, n_faces, fb = detect(frame, detect_height, upsample)
        fallbacks += fb
        if box is None:
            no_face += 1
            continue
        if n_faces > 1:
            multi_face += 1
        crop = crop_face(frame, *box)
        if crop.size == 0:
            no_face += 1
            continue
        # PNG round trip in memory: the authors read saved PNGs with cv2.imread(path, 0), whose
        # grayscale conversion differs from cvtColor by +-1 level. This reproduces it exactly.
        gray = cv2.imdecode(cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1], 0)
        crops.append(cv2.resize(gray, SIZE))
    cap.release()
    arr = np.stack(crops) if crops else np.zeros((0,) + SIZE, np.uint8)
    return video_path, arr, {"frames": frames, "faces": len(crops), "no_face": no_face, "multi_face": multi_face, "fallback": fallbacks}


def load_arcface(repo, weights, device):
    import torch
    sys.path.insert(0, repo)
    from models.resnet import resnet_face18
    model = resnet_face18(False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    return model.to(device).eval()


def embed(model, crops, device, batch_size=64):
    """crops: uint8 [n,128,128] -> float32 [n,1024]. Matches load_image + get_frame_id."""
    import torch
    out = []
    for i in range(0, len(crops), batch_size):
        b = crops[i:i + batch_size]
        pair = np.stack([b, b[:, :, ::-1]], axis=1).reshape(-1, 1, *SIZE)  # img, flip, img, flip, ...
        x = (pair.astype(np.float32) - 127.5) / 127.5
        with torch.no_grad():
            y = model(torch.from_numpy(x).to(device)).cpu().numpy()
        out.append(np.hstack((y[::2], y[1::2])))
    return np.concatenate(out) if out else np.zeros((0, 1024), np.float32)


def run_set(name, videos_dir, args, model, device, pool):
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    videos = sorted(glob(os.path.join(videos_dir, "*.mp4")) + glob(os.path.join(videos_dir, "*.avi")))
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        videos = videos[k::n]
    if args.limit:
        videos = videos[:args.limit]
    done_dirs = [out_dir] + [os.path.join(d, name) for d in args.skip_dirs]

    def is_done(v):
        f = os.path.splitext(os.path.basename(v))[0] + ".npy"
        return any(os.path.exists(os.path.join(d, f)) for d in done_dirs)

    todo = [v for v in videos if not is_done(v)]
    print(f"[{name}] {len(videos)} videos, {len(todo)} to do, from {videos_dir}", flush=True)
    import time
    t0, total_frames = time.time(), 0
    log_path = os.path.join(out_dir, "_log.csv")
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["video", "frames", "faces", "no_face", "multi_face", "fallback"])
        for n, (path, crops, st) in enumerate(pool.imap_unordered(process_video, [(v, args.detect_height, args.upsample) for v in todo]), 1):
            vid = os.path.splitext(os.path.basename(path))[0]
            ids = embed(model, crops, device)
            tmp = os.path.join(out_dir, vid + ".tmp.npy")
            np.save(tmp, ids.astype(np.float32))
            os.replace(tmp, os.path.join(out_dir, vid + ".npy"))
            w.writerow([vid, st["frames"], st["faces"], st["no_face"], st["multi_face"], st["fallback"]])
            f.flush()
            total_frames += st["frames"]
            if args.deadline and time.time() > args.deadline:
                print(f"[{name}] time budget reached after {n}/{len(todo)} videos; stopping cleanly", flush=True)
                return False
            if n % 20 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"[{name}] {n}/{len(todo)} videos, {total_frames} frames, {dt:.0f}s, "
                      f"{total_frames / max(dt, 1e-9):.1f} frames/s", flush=True)
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(here, "..", "TI2Net")
    p = argparse.ArgumentParser()
    p.add_argument("--ffpp_root")
    p.add_argument("--compression", default="c23")
    p.add_argument("--subsets", nargs="+", default=list(FFPP_SUBSETS))
    p.add_argument("--videos_dir")
    p.add_argument("--name")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--repo", default=repo)
    p.add_argument("--weights", default=os.path.join(repo, "pretrained", "resnet18_110.pth"))
    p.add_argument("--limit", type=int, default=0, help="only the first N videos of each set (for tests)")
    p.add_argument("--shard", default="", help="K/N: take every N-th video starting at K (parallel sessions)")
    p.add_argument("--skip_dirs", nargs="*", default=[], help="other output dirs whose finished videos are skipped")
    p.add_argument("--detect_height", type=int, default=0,
                   help="0 = authors' setting (detect on the full frame). >0 = detect on a copy shrunk to this height")
    p.add_argument("--upsample", type=int, default=1, help="dlib upsample count (authors use 1)")
    p.add_argument("--deadline", type=float, default=0, help="unix time after which no new video is started")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} workers={args.workers} detect_height={args.detect_height} upsample={args.upsample}", flush=True)
    model = load_arcface(args.repo, args.weights, device)

    with Pool(args.workers, initializer=_init_worker) as pool:
        if args.ffpp_root:
            for s in args.subsets:
                if not run_set(s, os.path.join(args.ffpp_root, FFPP_SUBSETS[s], args.compression, "videos"),
                        args, model, device, pool):
                    break
        elif args.videos_dir and args.name:
            run_set(args.name, args.videos_dir, args, model, device, pool)
        else:
            p.error("give --ffpp_root, or --videos_dir with --name")


if __name__ == "__main__":
    main()
