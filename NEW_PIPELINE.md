# New pipeline

Описание нового пайплайна для классификации.

## Download

1. Задать свой путь для папки data_path в файле `cfg/default.yaml`. Эта папка используется для хранения всех данных этого пайплайна.

2. Запустить скачивание датасета группы tits_size:
```
python classification/group_loader.py --group tits_size
```

3. Запустить скачивание датасета background:
```
python classification/json_loader.py --json_path ./classification/data/json2load.json
```

## Segmentation

1. Запустить сегментацию всех изображений в датасете. Вместо YOUR_MODEL_PATH вставить путь к вашей модели YOLO, отвечающей за данную сегментацию:
```
python classification/segmentation_builder.py --model_path YOUR_MODEL_PATH
```

## Train

1. Запуск тренировки производится с помощью файла `classification/train_wb.py`. Пример запуска для категории tits_size, с количеством эпох 60:
```
python classification/train_wb.py --cat tits_size --epochs 60
```




