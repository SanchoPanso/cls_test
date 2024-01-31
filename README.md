# CLS
## Документация

Документация в readthedocs: https://sanchopanso-organization-cls.readthedocs-hosted.com/en/latest/

## База данных

Запуск базы данных PostgreSQL с данными об изображениях:
```
cd postgres
docker-compose up
```

Пути:
- скрипты инициализации: `postgres/postgresql/init-db/`
- Папка с данными: `postgres/postgresql/data/`
- папка с дампами: `postgres/postgresql/dumps/`

Подключение к psql для отправки sql-запросов:
```bash
psql --username=psql_user --dbname=psql_db
```

Сделать дамп в файл `postgres/postgresql/dumps/dump.tar.gz` (путь в контейнере - `/dumps/dump.tar.gz`):
```bash
docker exec -it <postgres-container-id> /bin/bash
pg_dump -h 127.0.0.1 -U psql_user -F c -f /dumps/dump.tar.gz psql_db
pg_restore --clean -h 127.0.0.1 -U psql_user -F c -d psql_db /dumps/dump.tar.gz
```

Восстановить базу из дампа `postgres/postgresql/dumps/dump.tar.gz`:
```bash
docker exec -it <postgres-container-id> /bin/bash
pg_restore --clean -h 127.0.0.1 -U psql_user -F c -d psql_db /dumps/dump.tar.gz
```

