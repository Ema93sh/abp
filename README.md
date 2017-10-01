gcloud ml-engine local train --package-path abp \
                           --module-name abp.task_runner \
                           --distributed \
                           -- \
                           --example $EXAMPLE \
                           --adaptive $ADAPTIVE


# Cloud ML
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="abp_$now"
TRAINER_PACKAGE_PATH="/Users/Ema93sh/workspace/machine_learning/tensorflow/abp/abp"
MAIN_TRAINER_MODULE="abp.trainer.task_runner"
JOB_DIR="gs://abp-bucket/abp/"
MODEL_PATH="gs://abp-bucket/abp/saved_models/hra/tictactoe/v1/tictactoe.ckpt"
EXAMPLE="tictactoe"
ADAPTIVE="dqn"

gcloud ml-engine jobs submit training $JOB_NAME \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --job-dir $JOB_DIR \
    --region us-east1 \
    --stream-logs \
    -- \
    --example $EXAMPLE \
    --adaptive $ADAPTIVE \
    --model-path ./saved_models/hra/fruitcollection/fruitcollection.ckpt
    --training-episodes 50000
    --decay-steps 2000


gs://abp-bucket/job/abp/dqn/tensorflow_summaries

## Local
TRAINER_PACKAGE_PATH="/Users/Ema93sh/workspace/machine_learning/tensorflow/abp/abp"
MAIN_TRAINER_MODULE="abp.trainer.task_runner"
JOB_DIR="."
EXAMPLE="cartpole"
ADAPTIVE="dqn"

gcloud ml-engine local train \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --job-dir $JOB_DIR \
    -- \
    --example $EXAMPLE \
    --adaptive $ADAPTIVE \
    --no-render
