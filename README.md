# Cat Detection v2 — Unit 6 Final Assessment

## Image for leaderboard

```bash
docker pull aliagabalayev/cat-detector:final
```

**Image:** `aliagabalayev/cat-detector:final`  
**Student:** Ali Agabalazade

## Run

```bash
# Student info
docker run --rm aliagabalayev/cat-detector:final info

# Predict
docker run --rm \
  -v /absolute/path/to/images:/data/input:ro \
  -v /absolute/path/to/results:/data/output \
  aliagabalayev/cat-detector:final predict
```

> **Linux / SELinux (Fedora, RHEL):** append `:z` to volume flags if permission errors occur:
> ```bash
> docker run --rm \
>   -v /path/to/images:/data/input:ro,z \
>   -v /path/to/results:/data/output:z \
>   aliagabalayev/cat-detector:final predict
> ```

Output: `/data/output/predictions.csv` — `image_path,xmin,ymin,xmax,ymax,confidence,class`

## Model Results

| | Week-1 baseline | v2 (shipped) |
|---|---|---|
| Backbone | yolo26s | yolo26s |
| Epochs | 30 | 70 (10 frozen + 60 unfrozen) |
| mAP@0.5 | 0.9102 | 0.9217 |
| mAP@0.5:0.95 | 0.6974 | 0.7068 |

## Week-2 Techniques

Two-stage transfer learning · Cosine LR · copy_paste=0.3 · mixup=0.1 · scale=0.7 · Albumentations
