# New pipeline
Описание нового пайплайна для классификации.

## Download
1. Запустить скачивание датасета по группе. Например, скачать группу `tits_size`:
```
python classification/group_loader.py --group tits_size
```

2. Запустить скачивание датасета заднего фона `background`:
```
python classification/json_loader.py --json_path ./classification/data/json2load.json
```

## Segmentation
1. Запустить сегментацию всех изображений в датасете. Вместо YOUR_MODEL_PATH вставить путь к вашей модели YOLO, отвечающей за данную сегментацию:
```
python classification/segment_builder.py --model_path YOUR_MODEL_PATH
```

## Train
1. Запуск тренировки производится с помощью файла `classification/train_wb.py`. Пример запуска для категории tits_size, с количеством эпох 60:
```
python classification/train_wb.py --cat tits_size --epochs 60
```

## Mask building
1. Запуск создания масок по контурам:
```
python classification/inference.py --group tits_size
```

## Inference
1. Запуск инференса по группе (берет маски из контуров в папке `segments`):
```
python classification/inference.py --group test --model MODEL_PATH
```

2. Запуск инференса по папке:
```
python classification/inference.py --source DIR_PATH --model MODEL_PATH
```







