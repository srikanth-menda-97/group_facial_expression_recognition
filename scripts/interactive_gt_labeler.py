import os
import sys
import glob
import argparse
import pandas as pd
import cv2

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]
KEYMAP = {
    ord('1'): "angry",
    ord('2'): "disgust",
    ord('3'): "fear",
    ord('4'): "happy",
    ord('5'): "sad",
    ord('6'): "surprise",
    ord('7'): "neutral",
}

def load_existing_gt(gt_path):
    if not os.path.exists(gt_path):
        return {}
    df = pd.read_csv(gt_path)
    if "image" not in df.columns or "gt_label" not in df.columns:
        return {}
    return {str(r["image"]): str(r["gt_label"]).strip().lower() for _, r in df.iterrows()}

def save_gt(gt_path, gt_dict):
    rows = [{"image": k, "gt_label": v} for k, v in sorted(gt_dict.items())]
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(gt_path, index=False)

def draw_instructions(frame, image_name, existing=None):
    # Draw label instructions and current state on the image
    h = 22
    y0 = 20
    txts = [
        f"Image: {image_name}",
        "Choose label (keys): 1=angry 2=disgust 3=fear 4=happy 5=sad 6=surprise 7=neutral",
        "Controls: b=back  s=skip  r=reset-existing  q=quit",
    ]
    if existing:
        txts.append(f"Existing label: {existing} (press r to change)")
    for i, t in enumerate(txts):
        cv2.putText(frame, t, (10, y0 + i*h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

def console_prompt(image_name, existing=None):
    print(f"\nImage: {image_name}")
    if existing:
        print(f"Existing label: {existing} (press Enter to keep or type a new one from {EMOTIONS})")
    print("Enter one of:", ", ".join(EMOTIONS))
    print("Other commands: b=back, s=skip, q=quit")
    ans = input("> ").strip().lower()
    return ans

def main():
    ap = argparse.ArgumentParser(description="Interactive GT labeler: show each image and record gt_label to CSV.")
    ap.add_argument("--img-dir", default="data/group_images", help="Folder with images")
    ap.add_argument("--out", default="results/ground_truth.csv", help="Output CSV path")
    ap.add_argument("--resume", action="store_true", help="Resume from existing ground_truth.csv")
    ap.add_argument("--ext", nargs="+", default=[".jpg",".jpeg",".png"], help="Image extensions to include")
    args = ap.parse_args()

    # Gather images
    imgs = []
    for e in args.ext:
        imgs.extend(glob.glob(os.path.join(args.img_dir, f"*{e}")))
    imgs = sorted(imgs)
    if not imgs:
        print(f"No images found in {args.img_dir} with ext {args.ext}")
        sys.exit(1)

    # Load existing labels (for resume or to skip already-labeled files)
    gt = load_existing_gt(args.out) if args.resume else {}

    idx = 0
    # If resume, jump to first unlabeled image
    if args.resume:
        for i, p in enumerate(imgs):
            name = os.path.basename(p)
            if name not in gt or not gt[name]:
                idx = i
                break
        else:
            print("All images already labeled.")
            save_gt(args.out, gt)
            return

    while 0 <= idx < len(imgs):
        path = imgs[idx]
        name = os.path.basename(path)
        # Attempt GUI path first
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read {name}, skipping.")
                idx += 1
                continue
            disp = img.copy()
            existing = gt.get(name)
            draw_instructions(disp, name, existing=existing)
            cv2.imshow("Labeler", disp)
            # Wait indefinitely for a key press
            k = cv2.waitKey(0)  # returns key code
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            elif k == ord('s'):
                # skip without changing
                idx += 1
                continue
            elif k == ord('b'):
                # go back one image
                idx = max(0, idx - 1)
                continue
            elif k == ord('r') and existing:
                # reset existing so user can relabel
                gt.pop(name, None)
                continue
            elif k in KEYMAP:
                gt[name] = KEYMAP[k]
                save_gt(args.out, gt)  # save incrementally
                idx += 1
                continue
            else:
                # Unknown key — retry this image
                print(f"Unrecognized key: {k}. Valid: 1-7, b, s, r, q")
                continue
        except Exception:
            # Console fallback (e.g., headless environment)
            existing = gt.get(name)
            ans = console_prompt(name, existing=existing)
            if ans == "q":
                break
            elif ans == "s":
                idx += 1
                continue
            elif ans == "b":
                idx = max(0, idx - 1)
                continue
            elif ans == "" and existing:
                # keep existing
                idx += 1
                continue
            elif ans in EMOTIONS:
                gt[name] = ans
                save_gt(args.out, gt)
                idx += 1
                continue
            else:
                print(f"Invalid input. Use one of {EMOTIONS}, or b/s/q.")
                continue

    # Final save and cleanup
    save_gt(args.out, gt)
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print(f"Saved ground truth to {args.out} with {len(gt)} labeled images.")

if __name__ == "__main__":
    main()
