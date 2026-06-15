#!/usr/bin/env python3
"""
Cold-start 전용 평가 스크립트.

별도 학습 없이 기존 학습된 모델로 cold-start 성능 평가.
cold item = test ground truth 아이템 중 train_freq < cold_threshold 인 것.

Usage:
    python scripts/eval_coldstart.py \
        --dataset amazon_baby \
        --lm_model qwentext_8b \
        --model_version mlp \
        --n_layers 2 \
        --tau 0.2 \
        --train_norm --pred_norm \
        --neg_sample 256 --infonce 1 \
        --saveID my_run \
        --Ks 20 \
        --cold_threshold 5 \
        --cuda 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import collections
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


def compute_metrics(ranked_list: list[int], gt_set: set[int], K: int) -> dict:
    hits = [1 if item in gt_set else 0 for item in ranked_list[:K]]
    n_gt = min(len(gt_set), K)

    recall    = sum(hits) / len(gt_set) if gt_set else 0.0
    hit_ratio = 1.0 if sum(hits) > 0 else 0.0
    precision = sum(hits) / K

    dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_gt))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    mrr = 0.0
    for i, h in enumerate(hits):
        if h:
            mrr = 1.0 / (i + 1)
            break

    return dict(recall=recall, ndcg=ndcg, hit_ratio=hit_ratio,
                precision=precision, mrr=mrr)


def find_checkpoint(base_path: str) -> str:
    # exact path 먼저 시도
    ckpts = sorted(glob.glob(os.path.join(base_path, "*.pth.tar")))
    if not ckpts:
        ckpts = sorted(glob.glob(os.path.join(base_path, "*.pth")))

    # 없으면 날짜 prefix (MMDD_) 붙은 경로로 재시도
    if not ckpts:
        parent = os.path.dirname(base_path.rstrip("/"))
        dirname = os.path.basename(base_path.rstrip("/"))
        for candidate in sorted(glob.glob(os.path.join(parent, f"????_{dirname}"))):
            ckpts = sorted(glob.glob(os.path.join(candidate, "*.pth.tar")))
            if not ckpts:
                ckpts = sorted(glob.glob(os.path.join(candidate, "*.pth")))
            if ckpts:
                break

    if not ckpts:
        raise FileNotFoundError(f"checkpoint 없음: {base_path}")

    def epoch_num(p):
        try:
            return int(os.path.basename(p).split(".")[0].replace("epoch=", ""))
        except ValueError:
            return -1
    return max(ckpts, key=epoch_num)


def build_cold_items(data_path: str, dataset: str, threshold: int) -> set[int]:
    """train_freq < threshold 인 아이템을 cold item으로 반환."""
    train_file = os.path.join(data_path, dataset, "cf_data", "train.txt")
    train_freq: collections.Counter = collections.Counter()
    with open(train_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            for x in parts[1:]:
                train_freq[int(x)] += 1

    # valid + test에 등장하는 아이템만 대상
    eval_items: set[int] = set()
    for split in ["valid", "test"]:
        p = os.path.join(data_path, dataset, "cf_data", f"{split}.txt")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                for x in parts[1:]:
                    eval_items.add(int(x))

    cold = {i for i in eval_items if train_freq.get(i, 0) < threshold}
    return cold


def main():
    from parse import parse_args

    # cold_threshold를 sys.argv에서 직접 파싱 (parse_args 간섭 방지)
    cold_threshold = 5
    for i, v in enumerate(sys.argv):
        if v == "--cold_threshold" and i + 1 < len(sys.argv):
            cold_threshold = int(sys.argv[i + 1])

    args, special_args = parse_args()
    args.cold_threshold = cold_threshold

    print(f"\n{'='*55}")
    print(f"Cold-Start 평가  (threshold < {args.cold_threshold})")
    print(f"  dataset:  {args.dataset}")
    print(f"  model:    {args.model_name}  (lm_model: {getattr(args, 'lm_model', 'N/A')})")
    print(f"  layers:   {args.n_layers}  tau: {getattr(args, 'tau', 'N/A')}")
    print('='*55)

    # ── Data 로드 ──────────────────────────────────────────────────
    try:
        exec(f"from models.General.{args.model_name} import "
             f"{args.model_name}_Data as _DataCls", globals())
        data = eval("_DataCls(args)")
    except Exception:
        from models.General.base.abstract_data import AbstractData
        data = AbstractData(args)

    # ── cold item 정의 (threshold 기반) ────────────────────────────
    cold_item_ids = build_cold_items(args.data_path, args.dataset,
                                     args.cold_threshold)
    print(f"\n  cold items: {len(cold_item_ids)} "
          f"(train_freq < {args.cold_threshold})")

    # cold test set 구성
    cold_test: dict[int, list[int]] = {}
    for uid, gt_items in data.test_user_list.items():
        cold_gt = [i for i in gt_items if i in cold_item_ids]
        if cold_gt:
            cold_test[uid] = cold_gt

    total_test_users = len(data.test_user_list)
    print(f"  cold test users: {len(cold_test)} / {total_test_users} "
          f"({100*len(cold_test)/max(total_test_users,1):.1f}%)")

    if not cold_test:
        print("  ⚠️  cold test interaction이 없습니다.")
        return

    # ── Model 로드 ─────────────────────────────────────────────────
    running_model = (args.model_name + "_batch"
                     if (args.infonce == 1 and args.neg_sample == -1)
                     else args.model_name)
    exec(f"from models.General.{args.model_name} import {running_model} as _ModelCls",
         globals())
    model = eval("_ModelCls(args, data)")
    model.cuda(args.cuda)

    # ── checkpoint 로드 ────────────────────────────────────────────
    tn = "1" if args.train_norm else "0"
    pn = "1" if args.pred_norm else "0"
    saveID_str = (f"{args.saveID}_t{tn}p{pn}"
                  f"_Ks={args.Ks}_patience={args.patience}"
                  f"_n_layers={args.n_layers}_batch_size={args.batch_size}"
                  f"_neg_sample={args.neg_sample}_lr={args.lr}")
    for sa in special_args:
        saveID_str += f"_{sa}={getattr(args, sa)}"

    if args.model_name == 'LightGCN' and args.n_layers == 0:
        base_path = f"./weights/General/{args.dataset}/MF/{saveID_str}/"
    elif args.n_layers > 0 and args.model_name != 'LightGCN':
        base_path = f"./weights/General/{args.dataset}/{running_model}-LGN/{saveID_str}/"
    else:
        base_path = f"./weights/General/{args.dataset}/{running_model}/{saveID_str}/"

    ckpt_path = find_checkpoint(base_path)
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{args.cuda}")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  checkpoint: {ckpt_path}")

    # ── 추론 ──────────────────────────────────────────────────────
    K = args.Ks
    all_metrics = defaultdict(list)

    with torch.no_grad():
        all_users_emb, all_items_emb = model.compute()
        if args.pred_norm:
            all_users_emb = F.normalize(all_users_emb, dim=-1)
            all_items_emb = F.normalize(all_items_emb, dim=-1)

        for uid, cold_gt in cold_test.items():
            u_emb = all_users_emb[uid].unsqueeze(0)
            scores = torch.matmul(u_emb, all_items_emb.T).squeeze(0)

            train_items = data.train_user_list.get(uid, [])
            if train_items:
                scores[torch.tensor(train_items)] = -1e9

            _, top_k = torch.topk(scores, K)
            top_k = top_k.cpu().numpy().tolist()

            m = compute_metrics(top_k, set(cold_gt), K)
            for key, val in m.items():
                all_metrics[key].append(val)

    # ── 결과 출력 ─────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  dataset:          {args.dataset}")
    print(f"  model:            {args.model_name}  (lm_model: {getattr(args, 'lm_model', 'N/A')})")
    print(f"  cold_threshold:   < {args.cold_threshold}")
    print(f"  evaluated users:  {len(all_metrics['recall'])}")
    print(f"{'─'*50}")
    for metric, vals in all_metrics.items():
        print(f"  {metric:12s}@{K}: {np.mean(vals):.4f}")
    print(f"{'─'*50}\n")

    # ── CSV 저장 ───────────────────────────────────────────────────
    import csv
    out_dir = os.path.join("logs", "coldstart")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")
    write_header = not os.path.exists(out_csv)

    row = dict(
        dataset=args.dataset,
        model_name=args.model_name,
        lm_model=getattr(args, 'lm_model', 'N/A'),
        n_layers=args.n_layers,
        tau=getattr(args, 'tau', 'N/A'),
        model_version=getattr(args, 'model_version', 'N/A'),
        cold_threshold=args.cold_threshold,
        K=K,
        n_cold_users=len(all_metrics["recall"]),
        n_cold_items=len(cold_item_ids),
    )
    for metric, vals in all_metrics.items():
        row[metric] = f"{np.mean(vals):.6f}"

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  결과 저장: {out_csv}")


if __name__ == "__main__":
    main()
