import sqlite3

def create_table():
    with sqlite3.connect("experiment_results.db") as connection:
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algo TEXT,
            example_id INTEGER,
            alpha REAL,
            runtime INTEGER,
            true_val REAL,
            result REAL,
            iterations INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            IS_func TEXT,
            T INTEGER,
            K INTEGER,
            J INTEGER,
            m INTEGER,
            stop TEXT
        )
        """)
        connection.commit()

def insert_result(algo, example_id, alpha, runtime, true_val, result, iter, note):
    with sqlite3.connect("experiment_results.db") as connection:
        cursor = connection.cursor()
        cursor.execute("""
        INSERT INTO results (algo, example_id, alpha, runtime, true_val, result, iterations, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (algo, example_id, alpha, runtime, true_val, result, iter, note))
        connection.commit()

# Create the table (if not exists)
create_table()

