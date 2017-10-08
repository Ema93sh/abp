Adaptation Based Programming
=============================



# How to run

## Local Run

To run an example locally use the following command

```

python -m abp.trainer.task_runner

```

It takes the following arguments:

| Command Options          | Description        |
|------------------|--------------------|
|--example EXAMPLE | The example to run |
|--adaptive ADAPTIVE|The adaptive to use for the run|
|--job-dir JOB_DIR  | The location to write tensorflow summaries|
|--model-path MODEL_PATH |The location to save the model |
|--restore-model      | Restore the model instead of training a new model|
|-r, --render       |   Set if it should render the test episodes|
|--training-episodes TRAINING_EPISODES| Set the number of training episodes|
|--test-episodes TEST_EPISODES| Set the number of test episodes|
|--decay-steps DECAY_STEPS | Set the decay rate for exploration|


## Using CloudML
To run the job using cloud ML use the following commands

#### Setup Job Parameters
```
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="abp_tictactoe_$now"
TRAINER_PACKAGE_PATH="/path-to-abp/abp"
TRAINER_CONFIG_PATH="/path-to-abp/abp/trainer/cloudml-gpu.yml"
MAIN_TRAINER_MODULE="abp.trainer.task_runner"
JOB_DIR="gs://path-to-job-dir"
MODEL_PATH="gs://path-to-model"
EXAMPLE="tictactoe"
ADAPTIVE="hra"
TRAINING_EPISODES=1000
DECAY_STEPS=250

```

#### Submit the job
```
gcloud ml-engine jobs submit training $JOB_NAME \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --job-dir $JOB_DIR \
    --region us-east1 \
    --stream-logs \
    --config=$TRAINER_CONFIG_PATH \
    -- \
    --example $EXAMPLE \
    --adaptive $ADAPTIVE \
    --model-path $MODEL_PATH \
    --training-episodes $TRAINING_EPISODES \
    --decay-steps $DECAY_STEPS
```
