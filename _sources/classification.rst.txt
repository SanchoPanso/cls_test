Classification
=============

Для модели классификации рабочий процесс выглядит следющим образом:

- :ref:`download`
- :ref:`segment_build`
- :ref:`train`
- :ref:`upload`

Также модуль имеет дополнительную функциональность:

- :ref:`mask_build`
- :ref:`inference`
- :ref:`download_wandb`


Далее в пимерах в качестве группы используется `tits_size`. Группа может быть заменена на любую доступную.
Все команды в примерах запускаются из корневой папки репозитория. 
Структура файлов соответствует дефолтной из файла `cls/classification/cfg/default.yaml`.
Подробнее о конфигурации: :doc:`configuration`.

.. _download:

Скачивание данных
-----------------

Для дальнейшей работы необходимо скачать изображения и их метаданные. 
После скачивания изображения помещаются в папку `DATA/pictures`, метаданные в `DATA/meta`
и дополнительно в папке `DATA/datasets` создается csv файл, который используется для тренировки. 

#. Cкачивание датасета по группе:

.. code-block:: console
    
    python cls/classification/load_group.py --group tits_size

#. Cкачивание датасета заднего фона `background` из файла со списком пиксетов `cls/classification/data/json2load_background.json`:

.. code-block:: console

    python cls/classification/load_json.py --json_path cls/classification/data/json2load_background.json


.. _segment_build:

Создание сегментации
-----------------

Помимо изображений и их метаданных, для различных задач (в частности, тренировки) небходимо создать сегментационные маски
людей на изображениях. Полученные маски хранятся в базе данных PostgreSQL. Подробнее о формате хранимой сегментации: :doc:`segmentation_format`.

#. Скрипт `segment_builder.py` проходится по всем изображениям в папке `DATA/pictures`. Вместо `YOUR_MODEL_PATH` 
необходимо вставить путь к вашей модели YOLO, отвечающей за сегментацию людей::  
    
    python cls/classification/segment_builder.py --model_path YOUR_MODEL_PATH

По умолчанию скрипт помечает все изображения как 'unchecked' 
(то есть не проверенными разметчиками на наличие плохих масок) в базе данных, 
а также игнорирует уже обраотанные изображения.

#. Обработать все изображения в папке `DATA/pictures`::    
    python cls/classification/segment_builder.py --model_path YOUR_MODEL_PATH --process_all


#. Пометить как 'approved' ("проверены" разметчиками и не содеражат плохих масок)::
    
    python cls/classification/segment_builder.py --model_path YOUR_MODEL_PATH --mark_approved

#. Влючить бэдлист (текстовый файл со списком плохих изображений) 
в базу данных с информацией об изображениях::
    
    python cls/classification/push_badlist.py --badlist_path cls/classification/data/badlist_common.txt

Проверить базу данных можно с помощью блокнота `cls/classification/notebooks/check_database.ipynb`.


.. _train:

Тренировка модели
-----------------

После того, как изображения скачаны, csv файл датасета сформирован и сегментации изображений сгенерированы,
можно запускать тренировку.

Пример запуска тренировки на 1 эпохи::

    python cls/classification/train.py --cat tits_size --epochs 1

Чтобы получить более подробную информацию о параметрах, воспользуйтесь параметром `--help`::
    
    python cls/classification/train.py --help

Перед тем, как тренировка начнется, программа предложит предоставить ей API ключ от аккаунта W&B.
Это позволить вести логирование и выгрузку моделей на этот сервис.

После того, как обучение закончится, результат сохранится в папке `DATA/models/tits_size/{experiment_name}`.
В этой папке будет содержаться:

- checkpoints - папка с чекпоинтами в формате pytorch-lightning;
- csv_logs - папка с логом тренировки в формате csv;
- torchscripts - папка с моделью в формате torchscript;
- train_batches - папка с тренировочными батчами (для визуальной проверки);
- onnx - папка с о=моделью в формате onnx.

.. _upload:

Выгрузка метаданных
-----------------


#. Создание ret_meta.json. Модели для инференса берутся из конфигурации
 `cls/classification/cfg/default.yaml` как переменная `MODELS`.

Пример запуска для изображении из групп `sasha test` и `test`::

    python cls/classification/segment_meta_builder.py --groups 'sasha test' 'test'


#. Отправка ret_meta.json на сервер::

    python cls/classification/post_ret_meta.py --groups 'sasha test' 


.. _mask_build:

Создание масок
--------------

Создание масок в папке `MARKUP/masks` для разметчиков или инференса. 

Запуск создания масок для группы `test`:

.. code-block:: console

    python cls/classification/mask_bulder.py --group test

Ntrcn

.. _inference:

Инференс
--------

Результаты сохраняются в папке `TEST/inference`.

Запуск инференса по группе (берет маски из контуров в папке `segments`)::
    
    python cls/classification/inference.py --group test --model MODEL_PATH


Запуск инференса по папке c изображениями::
    
    python cls/classification/inference.py --source DIR_PATH --model MODEL_PATH

.. _download_wandb:

Скачать обученную модель из W&B
-------------------------------

Скачать обученную модель из wandb::

    python cls/classification/download_wandb.py --artifact_name ARTIFACT_NAME
