Начало работы с cегментацией
============================

В этом разделе описывается процесс подготовки и обучения модели YOLOv8 для задачи инстансной сегментации. 

.. note::

   Прежде чем использовать этот гайд, убедитель, 
   что у вас запущен model_storage из шага :ref:`model_storage`.


Установка
---------

Для установки проекта CLS, выполните следующие команды:

.. code-block:: bash

   $ git clone https://github.com/t1masavin/CLS.git -b develop
   $ cd CLS
   $ python3 -m venv venv
   $ source venv/bin/activate
   (venv) $ pip install poetry
   (venv) $ poetry install

Получение исходного датасета
-----------------------------

Для тренировки нужны исходные датасеты в формате COCO, которые собираются разметчиками.
В целях демонстрации скачаем и распакуем в папку `segmentation_data/initial_datasets` 
небольшой демонстрационный датасет из общего хранилища (пароль ssh - "password"):

.. code-block:: bash

    mkdir -p segmentation_data/initial_datasets
    scp -P 4000 user@0.0.0.0:/storage/other/segmentation_dataset.zip segmentation_data_test/initial_datasets/segmentation_dataset.zip
    unzip segmentation_data/initial_datasets/segmentation_dataset.zip -d segmentation_data/initial_datasets/

Подготовка датасета
-----------------------------

Предполагается, что датасет после разметки представляет собой один или несколько архивов с разметкой в COCO формате. 
Сперва эти  архивы необходимо правильно распарсить. Для этого все архивы для текущей итерации обучения надо
разместить в одной папке, например segmentation_data/archives. Затем запустить скрипт для их парсинга:

.. code-block:: bash

    python cls/instance_segmentation/parse_archives.py --input_dir <путь до папки с архивами> --output_dir <путь до папки с промежуточным датасетом> 

Предположим что --input_dir segmentation_data/archives, а --output_dir segmentation_data/initial_datasets

Создание датасета в формате YOLO
--------------------------------

Скрипт create_yolo_dataset.py предназначен для конвертации датасетов, аннотированных в формате COCO, 
в формат, совместимый с моделью YOLO. 
Это следующий шаг в подготовке данных для обучения.

Итак, мы имеем исходные данные в папке segmentation_data/initial_datasets. 
В этой папке для всех датасетов должна быть следующая структура (названия датасетов могут быть другими):

.. code-block:: bash
    
    - segmentation_data/initial_datasets/
        - dataset1/
            - images/
            - annotations/
                - instances_default.json
        - dataset2/
            - images/
            - annotations/
                - instances_default.json
        -...


Для того, чтобы прочитать все исходные датасеты и сложить подготовленный датасет в папку `segmentation_data/hds_person_04052024`, воспользуйтесь следующей командной:

.. code-block:: bash

    python cls/instance_segmentation/create_yolo_dataset.py --src_dir <путь до папки с распаршенным датасетом в COCO> --dst_dir  <путь до желаемой папки с итоговым датасетом> --existing_datasets  <путь до папки с данными предыдущей итерации обучения (если есть)>

В данном случае: 
--src_dir segmentation_data/initial_datasets;
--dst_dir segmentation_data/hds_person_25042024;
--existing_datasets segmentation_data/hds_person_04052024


В итоге должна получиться папка со следующей структурой:

.. code-block:: bash

    - segmentation_data/hds_050624/
        - train/
            - images/
            - labels/
        - valid/
            - images/
            - labels/
        - data.yaml


Обучение модели YOLOv8 на задачу сегментации
-------------------------------------------

Скрипт train_yolo_seg.py используется для обучения модели YOLOv8, 
используя подготовленный на предыдущем шаге датасет. 
Скрипт инициирует обучение модели с возможностью настройки различных параметров, 
таких как количество эпох, размер батча и скорость обучения.

.. note::

    Также для обучения предусмотрена возможность логирования результатов в W&B. 
    Для ее использования потребуется зарегистрироваться на сервисе https://wandb.ai/ и 
    предоставить api-ключ во время тренировки.


Обучим модель на подготовленных данных:

.. code-block:: bash

    python cls/instance_segmentation/train_yolo_seg.py --data segmentation_data/hds_person_04052024/data.yaml --model yolov8s-seg.pt --epochs 1

.. note::

    Здесь используется легковесная модель `yolov8s-seg.pt` и небольшое количество эпох 1.
    В реальности используется тяжелая модель `yolov8x-seg.pt` и количество эпох больше 100.

В итоге получится следующая папка:

.. code-block:: bash

    - segmntation_data/models/
        - train/
            - weights/
                - best.pt   # Лучшая модель, полученная во время тренировки
                - last.pt
            - ...


Экспорт модели YOLOv8 в формат для инференса
--------------------------------------------

Для того, чтобы обеспечить эффективную работу полученной сети и совместимость с Triton Inference Server, 
ее необходимо конвертировать в соответствующий формат.
Возьмем последнюю полученную модель и преобразуем ее, воспользуйтесь следующей командой:

.. code-block:: bash

    python cls/instance_segmentation/export_to_trt.py --src_path segmentation_data/models/train/weights/best.pt --dst_path segmentation_data/models/train/infer


После этого в segmentation_data/inference_models/instance_segmentation_model появятся модели в формате TensorRT 
и в формате ONNX для работы с Triton, файл с метаданными и подготовленные архивы для отправки в хранилище: 

.. code_block:: bash

    - instance_segmentation_model/
        - model.onnx
        - model.plan
        - meta.json
        - model_onnx.zip
        - model_trt.zip

Чтобы отправить trt-модель в общее хранилище, с присвоением версии (к примеру, 0.0.3), 
воспользуйтесь API от model storage:

.. code-block:: bash

    curl -X 'POST' \
    'http://localhost:8300/upload_new_version/' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'src_file=@segmentation_data/models/train/infer/model_trt.zip;type=application/x-zip-compressed' \
    -F 'model_name=detection' \
    -F 'model_version=0.0.3'

В ответ должно вернуться {"version": "0.0.3"}, что скажет об успешной доставке модели в хранилище.

.. note::

    В будущем планируется обернуть этот вызов в отдельный скрипт.


Что дальше?
-----------

Чтобы подробнее ознакомиться с возможностями этого модуля, обратитесь к :doc:`instance_segmentation`.



