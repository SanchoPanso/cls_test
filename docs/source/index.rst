Документация ML для HDSex.org
=============================

Здесь размещены руководства для различных ML решений, созданных для сайта HDSex.org. 

Обучение
--------

Обучение моделей детекции и сегментации осуществляется в рамках репозитория `CLS <https://github.com/t1masavin/CLS.git>`_. 

Проект CLS представляет собой инструмент для тренировки моделей классификации и сегментации на основе данных, 
собранных из сервиса `collect.moster`. Этот проект обеспечивает автоматизированный процесс 
загрузки, обработки, использования данных для обучения нейронных сетей, а также выгрузку результатов работы.

Перейдите в следующие разделы, чтобы начать обучение от подготовки данных до отправки модели на развертывание:

* :doc:`get_started_with_classification`
* :doc:`get_started_with_instance_segmentation`

Развертывание
-------------

Развертывание сетей детекции и классификации осуществляется 
с помощью репозиториев `model storage <https://github.com/SanchoPanso/model_storage.git>`_ и 
`modeler <https://bitbucket.org/luckytube/modeler/src/develop/>`_.

Перейдите в следующие разделы, чтобы узнать, 
как разворачивать обученные модели детекции и классификации для инференса
с использованием Triton Server:

* :doc:`model_storage_deployment`
* :doc:`modeler_deployment`


Содержание
----------

.. toctree::

   get_started_with_classification
   get_started_with_instance_segmentation
   model_storage_deployment
   modeler_deployment
   classification
   instance_segmentation
   demonstration
   configuration
   postgres

