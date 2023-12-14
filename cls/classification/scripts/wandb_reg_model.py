import wandb

ENTITY = 'art-team'
PROJECT_NAME = 'inference'

run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='inference')
artifact = run.use_artifact(f'{ENTITY}/model-registry/coco128_yolov8n:latest', type='model')
artifact_dir = artifact.download()
wandb.finish()

