Начало работы с классификацией
==============================

Допустим, у вас стоит задача тренировки сетей классификации на основе данных, 
собранных в сервисе collect.moster. 
На этой странице рассказывается, как подготовить данные, обучить модель 
и подготовить ее к разворачиванию для инференса.

.. note::

   Прежде чем использовать этот гайд, убедитель, 
   что у вас запущен model_storage из шага :ref:`model_storage`.

Установка
---------

Для установки проекта CLS, выполните следующие команды:

.. code-block:: bash

   git clone https://github.com/t1masavin/CLS.git -b develop
   cd CLS
   python3 -m venv venv
   source venv/bin/activate
   pip install poetry
   poetry config virtualenvs.in-project true
   poetry install .

Настройка подключения к PostgreSQL
----------------------------------

Для того, чтобы уметь классифицировать людей на фото, их нужно правильно вырезать из изображения. 
Поэтому проект CLS использует базу данных PostgreSQL для хранения данных о сегментации людей. 

Прежде чем начать работу, убедитесь, что у вас установлен и запущен PostgreSQL. 
Затем укажите параметры подключения в файле конфигурации проекта.
Подробнее см. :ref:`cls_postgres`.

.. note::

   Как правило, база данных уже развернута.
   Проверить это можно, запустив команду `docker ps`. 
   По умалчанию в списке контейнеров требуемая БД будет выглядеть примерно так:

   .. code-block :: bash

      7d5f83ae53dd   postgres:16.1-alpine3.19       "docker-entrypoint.s…"   3 months ago   Up 12 days    0.0.0.0:5432->5432/tcp, :::5432->5432/tcp     postgresql_postgres_1


Скачивание датасета
-------------------

Для обучения модели нужны изображения и их разметка. 
Допустим, мы хотим обучить классифицировать группу `tits_size`.
Для загрузки такого датасета из сервиса `collect.moster` выполните следующую команду:

.. code-block:: bash

   python cls/classification/load_group.py --group tits_size

После выполнения этой команды в корневой папке репозитория должна появиться 
папка classification_data со следующим содержанием:

.. code-block:: bash

   - classification_data/
      - pictures/
         - ...
      - meta/
         - ...
      - datasets/
         - tits_size.json

Что произошло? Создалась папка с данными `classification_data`, путь к которой прописан в файле конфигурации. 
В этой папке скачались изображения в папку `pictures`, скачались метаданные о наборах изображений в папку `meta`
и на их основе создан json-файл со сводной информацией о датасете. Подробнее см. :doc:`bottles`.


Сегментация людей
-----------------

После загрузки датасета можно приступить к сегментации людей на изображениях и 
записи сегментационных масок в базу данных.
Это необходимо, поскольку чтобы классифицировать человека, его нужно вырезать из фотографии. 

Скачаем обученную модель из общего хранилища и сохраним в `person_models/best.pt` 
следующими командами (пароль ssh - "password"):

.. code-block:: bash

    mkdir person_models
    scp -P 4000 user@0.0.0.0:/storage/other/best.pt person_models/best.pt


Теперь запустите сегментацию:

.. code-block:: bash

   python cls/classification/segment_builder.py --model_path person_models/best.pt

.. note::

   Если база данных уже развернута, скорее всего, картинки уже пройдены,
   процесс завершится быстро и в логе вы увидите следующее сообщение:

   .. code-block:: bash

      __main__: DB already has 37805 images, that will be skiped during segmentation
   
   Так и задумано. Если вы хотите перезаписать сегментацию, то используйте флаг `--process_all`
   

Чтобы обучить новую модель сегментции, см. :doc:`instance_segmentation`.


Тренировка классификации
------------------------

После получения сегментированных данных можно приступить к тренировке модели классификации. 
Запустите процесс обучения следующей командой:

.. code-block:: bash

   python cls/classification/train.py --cat tits_size --epochs 1 --batch 4

.. note:: 

   Это обучение всего лишь на одну эпоху и с маленьким батчем. 
   Оно сделано просто в качестве демонстрации.
   В реальности количество эпох обычно больше 50, а батч - больше 16.

После этого в папке classification_data/models/tits_size появится папка с прошедшим экспериментом, 
в которой можно найти логи и полученные модели:

.. code-block:: bash

   - DATA/models/tits_size/v__0_train_eff_16_0.001/
         - checkpoints/  
         - csv_logs/  
         - onnx/  
         - torchscripts/  
         - train_batches/

* checkpoints - папка с чекпоинтами в формате pytorch-lightning;
* csv_logs - папка с логом тренировки в формате csv;
* torchscripts - папка с моделью в формате torchscript;
* train_batches - папка с тренировочными батчами (для визуальной проверки);
* onnx - папка с моделью в формате onnx.

Конвертация в TensorRT
----------------------

.. attention::

   Конвертация модели может происходить некорректно, необходимо это проверить. 

Для эффективного деплоя модели нужно конвертировать ее в формат TensorRT.
Это можно сделать следующей командой:

.. code-block:: bash

   python cls/classification/export.py --group tits_size

Скрипт автоматически найдет последнюю обученную модель для группы tits_size и произведет конвертацию.
После выполнения появится папка со следующим содержанием:

.. code-block:: bash

    - classification_data/
        - inference_models/
            - tits_size/
                - inference_model/
                    -meta.json  
                    - model.onnx  
                    - model_onnx.zip    # версия модели для triton в формате onnx
                    - model.plan  
                    - model_trt.zip     # версия модели для triton в формате trt


Отправка модели в хранилище
---------------------------

Чтобы отправить trt-модель в общее хранилище, с присвоением версии (к примеру, 0.0.2), 
воспользуйтесь API от model storage:

.. code-block:: bash

    curl -X 'POST' \
    'http://localhost:8300/upload_new_version/' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'src_file=@classification_data/inference_models/tits_size/inference_model/model_trt.zip;type=application/x-zip-compressed' \
    -F 'model_name=tits_size' \
    -F 'model_version=0.0.2'

В ответ должен вернуться следующий ответ, что скажет об успешной доставке модели в хранилище:: 

   {"version": "0.0.2"}


.. note::

    В будущем планируется обернуть этот вызов в отдельный скрипт.


Что дальше?
-----------

Чтобы подробнее ознакомиться с возможностями этоо модуля, обратитесь к :doc:`classification`.
