import argparse
import logging

training_inputs = {'scaleTier': 'CUSTOM',
    'masterType': 'complex_model_m',
    'workerType': 'complex_model_m',
    'parameterServerType': 'large_model',
    'workerCount': 9,
    'parameterServerCount': 3,
    'packageUris': ['gs://my/trainer/path/package-0.0.0.tar.gz'],
    'pythonModule': 'trainer.task',
    'args': ['--arg1', 'value1', '--arg2', 'value2'],
    'region': 'us-central1',
    'jobDir': 'gs://my/training/job/directory',
    'runtimeVersion': '1.2'}

job_spec = {
            'jobId': my_job_name,
            }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-j',
        '--job-id',
        help = "Job ID to submit to cloudML",
        required = True)

    parser.add_argument(
        '--package-path',
        help = "Path to the package",
        required = True
    )

    parser.add_argument(
        '--config-path',
        help = "Path to the config file",
        required = True
    )


    # Args to the task runner
    parser.add_argument(
        '-e', '--example',
        help = 'The example to run',
        required = True
    )

    parser.add_argument(
        '-a', '--adaptive',
        help = 'The adaptive to use for the run',
        required = True
    )

    parser.add_argument(
        '--job-dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )

    parser.add_argument(
        '-mp', '--model-path',
        help = "The location to save the model"
    )

    parser.add_argument(
        '--restore-model',
        help = 'Restore the model instead of training a new model',
        action = 'store_true'
    )

    parser.add_argument(
        '--training-episodes',
        help = "Set the number of training episodes",
        type = int,
        default = 500
    )

    parser.add_argument(
        '--test-episodes',
        help = "Set the number of test episodes",
        type = int,
        default = 100
    )

    parser.add_argument(
        '--decay-steps',
        help = "Set the decay rate for exploration",
        type = int,
        default = 250
    )

    credentials = GoogleCredentials.get_application_default()
    cloudml = discovery.build('ml', 'v1', credentials=credentials)

    request = cloudml.projects().jobs().create(body=job_spec,
              parent=project_id)
    response = request.execute()

    try:
        response = request.execute()
        # You can put your code for handling success (if any) here.

    except errors.HttpError, err:
        # Do whatever error response is appropriate for your application.
        # For this example, just send some text to the logs.
        # You need to import logging for this to work.
        logging.error('There was an error creating the training job.'
                      ' Check the details:')
        logging.error(err._get_reason())



if __name__ == '__main__':
    main()
