Demonstration
=============

Демонстрация работы YOLO
-------------------------

Чтобы продемонстрировать работу обученной сети YOLOv8, перейдите в файл ``cls/demonstration/gradio_yolo.py``
и укажите путь до модели в формате .pt:

.. code-block:: python

    YOLO_PATH = "/data/achernikov/workspace/CLS/hds_persons_yolov8/train10/weights/best.pt"

Далее запустите скрипт в фоновом режиме:

.. code-block:: bash

    nohup python -u cls/demonstration/gradio_yolo.py > gradio_yolo.out & 

После этого лог программы будет записываться в файл ``gradio_yolo.out``.
В нем вы сможете найти публичную ссылку, по которой нужно перейти, 
чтобы начать тестирование:

.. code-block:: bash
    :emphasize-lines: 2

    Running on local URL:  http://127.0.0.1:7860
    Running on public URL: https://c3918ba8d689c8cf42.gradio.live

    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)

