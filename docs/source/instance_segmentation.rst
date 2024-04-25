Сегментация
============

`Instance Segmentation <https://paperswithcode.com/task/instance-segmentation>`_ - 
задача компьютерного зрения, целью которой является нахождение отдельных объектов на изображении, 
которое включет в себя определение границ каждого объекта, присвоение каждому объекту метки класса и 
определение попиксельной маски объекта.

В этом разделе описывается процесс подготовки и обучения модели YOLOv8 для задачи инстансной сегментации. 
Он включает в себя описание следующих шагов:

* Создание датасета в формате YOLO
* Обучение модели YOLOv8 на задачу сегментации
* Экспорт модели YOLOv8 в формат для инференса


Создание датасета в формате YOLO
--------------------------------

Скрипт create_yolo_dataset.py предназначен для конвертации датасетов, аннотированных в формате COCO, 
в формат, совместимый с моделью YOLO. 
Это первый шаг в подготовке данных для обучения.

Перед использованием скрипта, убедитесь, что исходные датасеты размещены в папке segmentation_data/source_datasets. 
В этой папке должна быть следующая структура:

.. code-block:: bash
    
    - segmentation_data/source_datasets/
        - dataset1/
            - images/
            - annotations/
                - instances_default.json
        - dataset2/
            - images/
            - annotations/
                - instances_default.json
        -...


Скрипт анализирует исходные аннотации COCO, извлекает информацию о положении объектов (bounding boxes)
 и их классификации, после чего преобразует её в нужный формат.

Использование:

.. code-block:: bash

    python create_yolo_dataset.py --src_dir <path_to_coco_annotations> --dst_dir <path_to_yolo_dataset>

Параметры командной строки:

* --src_dir: путь к папке с исходными датасетами в формате COCO.
* --dst_dir: путь к папке для сохранения датасета в формате YOLO.

Для примера возьмем датасеты из папки `segmentation_data/source_datasets` и преобразуем их в датасет YOLO в папке `segmentation_data/prepared_datasets/yolo_dataset`:

.. code-block:: bash

    python create_yolo_dataset.py --src_dir segmentation_data/source_datasets --dst_dir segmentation_data/prepared_datasets/yolo_dataset


В итоге должна получиться папка со следующей структурой:

.. code-block:: bash

    - segmentation_data/prepared_datasets/yolo_dataset/
        - train/
            - images/
            - labels/
        - valid/
            - images/
            - labels/
        - data.yaml


Обучение модели YOLOv8 на задачу сегментации
-------------------------------------------

Скрипт train_yolo_seg.py используется для обучения модели YOLOv8, используя подготовленный на предыдущем шаге датасет. 
Этот процесс является вторым шагом в подготовке модели к работе по задаче инстансной сегментации.

Скрипт инициирует обучение модели с возможностью настройки различных параметров, 
таких как количество эпох, размер батча и скорость обучения.

.. note::

    Также для обучения предусмотрена возможность логирования результатов в W&B. 
    Для этого потребуется зарегистрироваться на сервисе https://wandb.ai/ и предоставить api-ключ во время тренировки.


Использование:

.. code-block:: bash

    python train_yolo_seg.py --model <path_to_model> --project <name_of_project> --data <path_to_yolo_dataset> --epochs <num_epochs> --batch_size <batch_size>

Параметры командной строки:

* --model: путь к обучаемой модели.
* --project: название проекта (относительный путь к папке с прогонами)
* --data: путь к подготовленному датасету в формате YOLO.
* --epochs: количество эпох для обучения.
* --batch-size: размер батча.

.. note::

    Проверьте совместимость настроек обучения с вашей аппаратной конфигурацией, 
    особенно при использовании GPU, чтобы избежать проблем с переполнением памяти.

Для примера возмем полученный YOLO датасет в папке `segmentation_data/prepared_datasets/yolo_dataset` и обучим на нем model `yolov8x-seg.pt`.
Количество эпох поставим равным 1, чтобы быстрее увидеть результат. В реальной ситуации количество эпох обычно начинается от 50.
Остальные параметры оставим по умолчанию.

.. code-block:: bash

    python train_yolo_seg.py --data segmentation_data/prepared_datasets/yolo_dataset/data.yaml --model yolov8x-seg.pt


.. note::

    Для тренировки используется библиотека ultralytics. 
    Поэтому при необходимости более тонкой настройки параметров, стоит обратиться к ней.


После того, как тренировка закончилась, появится папка `yolov8_seg_runs`, в которой будут лежать результаты тренировки.
Помимо прочих результатов, в папке с прогоном можно найти натренированную модель `best.pt`.

.. code-block:: bash

    - yolov8_seg_runs/
        - train/
            - weights/
                - best.pt
                - last.pt
            - ...


Экспорт модели YOLOv8 в формат для инференса
--------------------------------------------

Для того, чтобы обеспечить эффективную работу полученной сети и совместимость с Triton Inference Server, 
ее необходимо конвертировать в соответствующий формат.
Возьмем полученную модель `best.pt` и преобразуем ее, воспользуйтесь следующей командой:

.. code-block:: bash

    python export_to_trt.py --src_path yolov8_seg_runs/train/weights/best.pt --dst_path segmentation_data/inference_models/new_model


После этого в segmentation_data/inference_models/new_model появится папки с моделью в формате TensorRT 
и в формате ONNX с файлами конфигурации для работы с Triton: 

.. code_block:: bash

    - new_model/
        - model_onnx/
            - 1/
                - model.onnx
                - meta.json
            - config.pbtxt

        - model_trt/
            - 1/
                - model.plan
                - meta.json
            - config.pbtxt


Папки `model_onnx` и `model_trt` - модели для инференса для репозитория моделей в Triton Server. 
Наиболее оптимальной является `model_trt` и она используется по умолчанию.
Для ее запуска разместите ее в репозитории моделей.


