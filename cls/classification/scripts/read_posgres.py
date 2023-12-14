import psycopg2

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
    cursor.execute("SELECT * FROM pictures")

    rows = cursor.fetchall()
    for row in rows:
        print(row)

connection.close()
