import argparse
from pathlib import Path
import random
import shutil
from tqdm import tqdm


def find_pairs(s1_path: Path, s2_path: Path, pair_suffix=('s1', 's2')):
    exts = ['*.png', '*.jpg', '*.jpeg']
    pairs = []
    s2_files = {p.name: p for ext in exts for p in s2_path.glob(ext)}
    for ext in exts:
        for s1_file in s1_path.glob(ext):
            # try exact name in s2
            cand = s2_files.get(s1_file.name)
            if cand and cand.exists():
                pairs.append((s1_file, cand))
                continue
            # try replace suffix pattern _s1_ -> _s2_
            replaced = s1_file.name.replace(f'_{pair_suffix[0]}_', f'_{pair_suffix[1]}_')
            cand2 = s2_path / replaced
            if cand2.exists():
                pairs.append((s1_file, cand2))
                continue
            # fallback: search by stem
            for p in s2_path.glob('*'):
                if p.stem == s1_file.stem:
                    pairs.append((s1_file, p))
                    break
    return pairs


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def split_dataset(src: Path, dst: Path, val_ratio=0.05, seed=42, pair_suffix=('s1', 's2')):
    random.seed(seed)
    src = Path(src)
    dst = Path(dst)
    assert src.exists(), f"src {src} not found"

    # detect if root contains pair dirs or category dirs
    # case A: src/s1 and src/s2
    if (src / pair_suffix[0]).exists() and (src / pair_suffix[1]).exists():
        categories = ['.']
    else:
        categories = [p.name for p in src.iterdir() if p.is_dir()]

    total_pairs = 0
    val_pairs = 0

    for cat in categories:
        if cat == '.':
            s1_path = src / pair_suffix[0]
            s2_path = src / pair_suffix[1]
            out_train_s1 = dst / 'train' / pair_suffix[0]
            out_train_s2 = dst / 'train' / pair_suffix[1]
            out_val_s1 = dst / 'val' / pair_suffix[0]
            out_val_s2 = dst / 'val' / pair_suffix[1]
        else:
            s1_path = src / cat / pair_suffix[0]
            s2_path = src / cat / pair_suffix[1]
            out_train_s1 = dst / 'train' / cat / pair_suffix[0]
            out_train_s2 = dst / 'train' / cat / pair_suffix[1]
            out_val_s1 = dst / 'val' / cat / pair_suffix[0]
            out_val_s2 = dst / 'val' / cat / pair_suffix[1]

        if not s1_path.exists() or not s2_path.exists():
            print(f"Skipping category {cat}: missing s1 or s2")
            continue

        pairs = find_pairs(s1_path, s2_path, pair_suffix=pair_suffix)
        total_pairs += len(pairs)
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        val = pairs[:n_val]
        train = pairs[n_val:]

        # ensure dirs
        for p in [out_train_s1, out_train_s2, out_val_s1, out_val_s2]:
            ensure_dir(p)

        # copy val
        for a, b in tqdm(val, desc=f"copy val {cat}"):
            shutil.copy2(a, out_val_s1 / a.name)
            shutil.copy2(b, out_val_s2 / b.name)
            val_pairs += 1

        # copy train
        for a, b in tqdm(train, desc=f"copy train {cat}"):
            shutil.copy2(a, out_train_s1 / a.name)
            shutil.copy2(b, out_train_s2 / b.name)

    print(f"Total pairs found: {total_pairs}, val pairs: {val_pairs}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=str, default="/home/lx/deep_learning/dataset/archive/v_2", help='source dataset root')
    p.add_argument('--dst', type=str, default="/home/lx/deep_learning/dataset/archive/v_2_split", help='destination split root')
    p.add_argument('--val_ratio', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pair_suffix', nargs=2, default=['s1', 's2'])
    p.add_argument('--move', action='store_true', help='move validation files from source to dst')
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.parent / f"{src.name}_split"
    print(f"src: {src}\ndst: {dst}\nval_ratio: {args.val_ratio}\nmove: {args.move}")
    split_dataset(src, dst, val_ratio=args.val_ratio, seed=args.seed, pair_suffix=tuple(args.pair_suffix))

    if args.move:
        print("Moving validation files from source to destination...")
        random.seed(args.seed)
        if (src / args.pair_suffix[0]).exists() and (src / args.pair_suffix[1]).exists():
            categories = ['.']
        else:
            categories = [p.name for p in src.iterdir() if p.is_dir()]

        for cat in categories:
            if cat == '.':
                s1_path = src / args.pair_suffix[0]
                s2_path = src / args.pair_suffix[1]
                out_val_s1 = dst / 'val' / args.pair_suffix[0]
                out_val_s2 = dst / 'val' / args.pair_suffix[1]
            else:
                s1_path = src / cat / args.pair_suffix[0]
                s2_path = src / cat / args.pair_suffix[1]
                out_val_s1 = dst / 'val' / cat / args.pair_suffix[0]
                out_val_s2 = dst / 'val' / cat / args.pair_suffix[1]

            if not s1_path.exists() or not s2_path.exists():
                continue

            pairs = find_pairs(s1_path, s2_path, pair_suffix=tuple(args.pair_suffix))
            random.shuffle(pairs)
            n_val = max(1, int(len(pairs) * args.val_ratio))
            val = pairs[:n_val]

            for a, b in tqdm(val, desc=f"move val {cat}"):
                ensure_dir(out_val_s1)
                ensure_dir(out_val_s2)
                dst_a = out_val_s1 / a.name
                dst_b = out_val_s2 / b.name
                if dst_a.exists() and dst_b.exists():
                    try:
                        if a.exists():
                            a.unlink()
                        if b.exists():
                            b.unlink()
                    except Exception as e:
                        print(f"Failed to remove original files: {e}")
                    continue
                try:
                    shutil.move(str(a), str(dst_a))
                    shutil.move(str(b), str(dst_b))
                except Exception as e:
                    print(f"Move failed for {a} or {b}: {e}")


if __name__ == '__main__':
    main()
