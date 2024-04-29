Добро пожаловать в документацию проекта CLS
=============================================

Проект CLS представляет собой инструмент для тренировки моделей классификации и сегментации на основе данных, 
собранных из сервиса `collect.moster`. Этот проект обеспечивает автоматизированный процесс 
загрузки, обработки, использования данных для обучения нейронных сетей, а также выгрузку результатов работы.

Установка
---------

Для установки проекта CLS, выполните следующие команды:

.. code-block:: bash

   $ git clone https://github.com/t1masavin/CLS.git -b develop
   $ cd CLS
   $ python3 -m venv venv
   $ source venv/bin/activate
   (venv) $ pip install poetry
   (venv) $ poetry install .

Начало работы
-------------

Чтобы начать использовать проект CLS, следуйте инструкциям 
в разделах :doc:`get_started_with_classification` и :doc:`get_started_with_instance_segmentation`.

Содержание
----------

.. toctree::

   get_started_with_classification
   get_started_with_instance_segmentation
   classification
   instance_segmentation
   demonstration
   configuration
   postgres

