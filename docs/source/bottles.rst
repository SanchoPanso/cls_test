Бутылки данных
==============

Описание
---------

Для того чтобы эффективно организовать проект по скачиванию изображений и обучению нейросетей для классификации, 
важно правильно структурировать хранение данных. 
В данной документации представлен раздел, посвященный концепции "бутылки данных" — специальной директории, 
которая служит контейнером для всех необходимых компонентов проекта: исходных данных, метаданных, обученных моделей и т.д. 
Эта структура помогает обеспечить порядок, удобство доступа и переносимость данных.

Бутылка данных — это директория, которая функционирует как самодостаточная единица для хранения всех аспектов проекта 
по машинному обучению. Этот подход позволяет легко управлять данными, перемещать их между различными средами 
или даже разными пользователями.

По умолчанию, бутылка данных располагается в папке `DATA` относительно корня репозитория (см. :doc:`configuration`).

Структура бутылки данных
-----------------------

Каждая бутылка данных содержит следующие компоненты:

    Исходные данные:
        Подпапка /pictures
        Содержит изображения, скачанные из различных источников.

    Метаданные:
        Подпапка /meta
        Включает файлы, описывающие исходные данные (например, источники данных, авторские права, описательные теги).

    Датасеты:
        Подпапка /datasets
        Включает json-файлы, описывающие датасеты

    Обученные модели:
        Подпапка /models
        Содержит модели нейросетей, обученные на основе исходных данных, включая файлы конфигурации и весов.

Преимущества использования
--------------------------

* Модульность: Бутылки данных могут быть независимо модифицированы, заменены или обновлены без влияния на другие части проекта.
* Переносимость: Все необходимые данные и модели упакованы в одну директорию, что облегчает перенос между машинами или средами.
* Воспроизводимость: Легко воспроизводить результаты и эксперименты, используя определенную версию бутылки данных.
* Управление версиями: Легко отслеживать версии данных и моделей, сохраняя предыдущие состояния проекта для анализа или отладки.
