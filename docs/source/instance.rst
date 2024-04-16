Instance
========

1. Выбираем в тулзе нуждные нам guids пиксетов и сохраняем их в файле CLS/classification/scripts/loaders/json2load.json
2. Запускаем CLS/classification/scripts/loaders/loader_from_json.py, он скачает контент по этим пиксетам и сохранит их в DATA meta|pictre|datasets
3. Запускаем TAGGING/CLS/classification/notes/segmentation_builder.py он пройдется по всем ранее не обработанным пикчам и сделает из них маски, с метой о bbox и классе
4. Запускаем CLS/classification/notes/segment_meta_builder.py он создаст ret_meta.json, нужно указать в скрипте, какую категорию, и какими сетками мы проверяем
5. Запускаем CLS/classification/scripts/post_retMeta.py с указанием группы, мету которой мы отправляем обратно

.. note::

    Эта страница пока просто заглушка

