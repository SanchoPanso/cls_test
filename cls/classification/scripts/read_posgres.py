import psycopg2
import json

try:
    connection = psycopg2.connect(
        host="localhost",
        database="psql_db",
        user="psql_user",
        password="root",
        port="5432"  # optional
    )
    print("Connection to PostgreSQL DB successful")
except psycopg2.OperationalError as e:
    print(f"Error: {e}")
    exit(1)

with connection.cursor() as cursor:
    cursor.execute("SELECT * FROM pictures WHERE path = \'00008c174d21b1e9504483e5128d756f4b0c.jpeg\'")

    rows = cursor.fetchall()
    for row in rows:
        print(row)
        segments = row[-1]
        print(type(segments))
        break
    
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N\'pictures\'")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

connection.close()
