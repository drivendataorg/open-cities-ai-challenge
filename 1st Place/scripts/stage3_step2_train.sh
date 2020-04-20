# 5 folds Unet model with different encoders
python -m src.train --config=configs/stage3-effb1-f0.yaml
python -m src.train --config=configs/stage3-srx50-f0.yaml
python -m src.train --config=configs/stage3-srx50-2-f0.yaml
python -m src.train --config=configs/stage3-inrv2-f0.yaml
python -m src.train --config=configs/stage3-effb4-f0.yaml
