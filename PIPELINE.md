TRAIN  
1 качаем контент по выбранной группе: CLS/classification/scripts/loaders/loader_by_group.py в файле сказать какую группу качать
2 из директории CLS/classification запускаем nohup $(pwd)/train.sh & предварительно в train.sh настоить конфигурацию обучения сети
3 модель будет в DATA/models, для её запуска torch.jit.load('DATA/models/model.pt', _extra_files=extra_files), где extra_files = {'num2label.txt': ''} - это файл с метками классов, если он есть

INFERENSE  
1 выбираем в тулзе нуждные нам guids пиксетов и сохраняем их в файле CLS/classification/scripts/loaders/json2load.json
2 запускаем CLS/classification/scripts/loaders/loader_from_json.py, он скачает контент по этим пиксетам и сохранит их в DATA meta|pictre|datasets
3 запускаем TAGGING/CLS/classification/notes/segmentation_builder.py он пройдется по всем ранее не обработанным пикчам и сделает из них маски, с метой о bbox и классе
4 запускаем CLS/classification/notes/segment_meta_builder.py он создаст ret_meta.json, нужно указать в скрипте, какую категорию, и какими сетками мы проверяем
5 запускаем CLS/classification/scripts/post_retMeta.py с указанием группы, мету которой мы отправляем обратно

DEMONSTARION  
1 запускаем CLS/demonstration/gradio_YOLO.py с указанием группы, мету которой мы отправляем обратно
