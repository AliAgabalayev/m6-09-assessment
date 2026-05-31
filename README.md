# Cat Detection v2 — Unit 6 Final Assessment

## Image for leaderboard

```bash
docker pull aliagabalayev/agabalazade-ali-cat-detector:submission
```

**Image:** `aliagabalayev/agabalazade-ali-cat-detector:submission`  
**Student:** Ali Agabalazade

## Run

```bash
docker run --rm \
  -v /path/to/images:/data/input:ro \
  -v /path/to/results:/data/output \
  aliagabalayev/agabalazade-ali-cat-detector:submission \
  --input /data/input \
  --output /data/output/predictions.json \
  --threshold 0.25
```

> **Linux / SELinux (Fedora, RHEL):** append `:z` to volume flags if permission errors occur.

Output: `/data/output/predictions.json` — JSON with `model`, `threshold`, `predictions[]`

## Model Results

| | Week-1 baseline | v2 (shipped) |
|---|---|---|
| Backbone | yolo26s | yolo26s |
| Epochs | 30 | 70 (10 frozen + 60 unfrozen) |
| mAP@0.5 | 0.9102 | 0.9217 |
| mAP@0.5:0.95 | 0.6974 | 0.7068 |

## Week-2 Techniques

Two-stage transfer learning · Cosine LR · copy_paste=0.3 · mixup=0.1 · scale=0.7 · Albumentations
