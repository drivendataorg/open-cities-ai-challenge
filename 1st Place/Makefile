UID := $(shell id -u)
GID := $(shell id -g)

# Execute test data preparation and prediction (if you have trained models in models/ directory)
inference: stage0 stage3-step3

# Execute data preparation, training and prediction for all stages from scratch
train-inference: stage0 stage1 stage2 stage3

# Build docker image with required environment
build:
	docker build -t open-cities:0.1 --build-arg "USER_UID=${UID}" .

# Start development container with docker compose in deamon mode
start:
	docker-compose up -d

# Install dependencies inside container
install:
	docker exec open-cities-dev pip install -r requirements.txt --user

# Stop container
clean:
	docker-compose down

# Prepare test data
stage0:
	docker exec open-cities-dev bash scripts/stage0_step0_prepare.sh

# Execute stage 1 (data preparation, models training, prediction)
stage1: stage1-step1 stage1-step2 stage1-step3

# Extract initial data, reproject, generate raster masks from vector geojson data
stage1-step1:
	docker exec open-cities-dev bash scripts/stage1_step1_prepare.sh

# Train 10 models on prepared data
stage1-step2:
	docker exec open-cities-dev bash scripts/stage1_step2_train.sh

# Make prediction for test data
stage1-step3:
	docker exec open-cities-dev bash scripts/stage1_step3_predict.sh


# Execute stage 2 (data preparation, models training, prediction)
stage2: stage2-step1 stage2-step2 stage2-step3

# Cut predictions from stage 1 and add them to training data (so called pseudo labeling)
stage2-step1:
	docker exec open-cities-dev bash scripts/stage2_step1_prepare.sh

# Train 10 models on prepared data (tier 1 + test predictions from stage 1)
stage2-step2:
	docker exec open-cities-dev bash scripts/stage2_step2_train.sh

# Make prediction for test data
stage2-step3:
	docker exec open-cities-dev bash scripts/stage2_step3_predict.sh


# Execute stage 3 (data preparation, models training, prediction)
stage3: stage3-step1 stage3-step2 stage3-step3

# Cut predictions from stage 2 and add them to training data (so called pseudo labeling)
stage3-step1:
	docker exec open-cities-dev bash scripts/stage3_step1_prepare.sh

# Train 5 models on prepared data (tier 1 + test predictions from stage 2)
stage3-step2:
	docker exec open-cities-dev bash scripts/stage3_step2_train.sh

# Make prediction for test data
stage3-step3:
	docker exec open-cities-dev bash scripts/stage3_step3_predict.sh
