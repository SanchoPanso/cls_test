PostgreSQL
=============

Для того, чтобы сегментационные маски были доступны для всех пользователей, 
создается отдельный контейнер с образом базы данных PostgreSQL.
Структура БД задается следующим образом:

.. code-block:: bash

    CREATE TABLE pictures (
        id SERIAL PRIMARY KEY,
        path VARCHAR(255) NOT NULL,
        model_version VARCHAR(50) NOT NULL,
        status VARCHAR(50) NOT NULL,
        segments jsonb DEFAULT NULL
    );


Настройка подключения
---------------------

1. Создать контейнер.
Если вы работаете с уже существующей БД, то переходите к п.2. 
Создание таблицы и процесс запуска контейнера уже реализован. 
Все, что нужно - это запустить docker compose:

.. code-block:: bash

    cd postgres
    docker compose up -d


2. После того, как БД развернута, нужно указать URL к БД в файле конфигурации по схеме: '{dialect}+{driver}://{user}:{password}@{host}:{port}/{database}. 
О файле конфигурации см. :doc:`configuration`.
Пример:

.. code-block:: bash

    pictures_info_db_url: 'postgresql+psycopg2://psql_user:root@172.20.0.1:5432/psql_db'


Резервное копирование
---------------------

Дописать...
