Формат сегментации
-----------------

Для начала опишем формат сегментации. 
В папке segments лежат json файлы, чьи имена совпадают 
с именами соответствующих изображений. 
Вот пример содержания такого json - файла::

    {
    "0": {
        "cls": "female",
        "status": "approved",
        "conf": 0.9656772017478943,
        "bbox": [
        7.10723876953125,
        3.7406158447265625,
        1598.975830078125,
        1042.731201171875
        ],
        "segments": [
        [
            0.4703125,
            0.29910714285714285,
            0.471875,
            0.3013392857142857,
            0.471875,
            0.30580357142857145,
            0.4734375,
            0.30580357142857145,
            0.475,
            0.3080357142857143,
            0.4765625,
            0.3080357142857143,
            0.475,
            0.3080357142857143,
            0.471875,
            0.30357142857142855,
            0.471875,
            0.29910714285714285
        ],
        [
            0.440625,
            0.2700892857142857
        ],]
    "1": ...,
    ...}
    }

В этом отображении ключи - порядковый номер объекта, 
начиная с нуля (объекты - люди, в данном случае), значения - их описание. В каждой секции объекта:

    * "cls" - имя класса ("female" или "male")
    * "status" - статус проверки сегментации (deprecated)
    * "conf" - уверенность сети
    * "bbox" - xywh (абсолютные координаты)
    * "segments" - список сегментов xyn (аналогично segmentation в COCO, но нормированный)

В ближайшем будущем планируется удалить "status" и сделать "bbox" нормированным.

