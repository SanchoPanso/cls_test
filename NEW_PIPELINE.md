# Новый пайплайн
Описание нового пайплайна для классификации.

## Основные функции
Это список необходимых действий для запуска тренировки.

1. Cкачивание датасета по группе. Например, скачать группу `tits_size`:
```
python classification/group_loader.py --group tits_size
```

2. Cкачивание датасета заднего фона `background` из файла со списком пиксетов `./classification/data/json2load.json`:
```
python classification/json_loader.py --json_path ./classification/data/json2load.json
```

3. Сегментация изображений в датасете. Вместо YOUR_MODEL_PATH вставить путь к вашей модели YOLO, отвечающей за данную сегментацию:
```
python classification/segment_builder.py --model_path YOUR_MODEL_PATH
```

4. Запуск тренировки. Пример запуска для категории tits_size, с количеством эпох 60:
```
python classification/train_wb.py --cat tits_size --epochs 60
```

## Дополнительные функции

1. Запуск создания масок для группы `tits_size`:
```
python classification/inference.py --group tits_size
```

2. Запуск инференса по группе (берет маски из контуров в папке `segments`):
```
python classification/inference.py --group test --model MODEL_PATH
```

2. Запуск инференса по папке c изображениями:
```
python classification/inference.py --source DIR_PATH --model MODEL_PATH
```







