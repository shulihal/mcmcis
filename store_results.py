import sqlite3

def create_table():
    with sqlite3.connect("data/experiment_results.db") as connection:
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algo TEXT,
            example_id INTEGER,
            true_val REAL,
            result REAL,
            beta REAL,
            gamma INTEGER,
            adaptive BOOL,
            pi REAL,
            window INTEGER,
            fraction REAL,
            IS_func TEXT,
            accept_rate REAL,
            up_rate REAL,
            pi_hat REAL,
            runtime INTEGER,
            iterations INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            T INTEGER,
            K INTEGER,
            J INTEGER,
            notes TEXT
        )
        """)
        connection.commit()

def insert_result(algo, example_id, true_val, result, 
                  beta, gamma, adaptive, pi, window, frac, IS_func,
                  accept_rate, up_rate, pi_hat, runtime, iterations, 
                  T, K, J, notes=None):     
    with sqlite3.connect("data/experiment_results.db") as connection:
        cursor = connection.cursor()
        cursor.execute("""
        INSERT INTO results (algo, example_id, true_val, result, 
                  beta, gamma, adaptive, pi, window, fraction, IS_func,
                  accept_rate, up_rate, pi_hat, runtime, iterations, 
                  T, K, J, notes)
        VALUES (?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?)
        """,(algo, example_id, true_val, result, 
                  beta, gamma, adaptive, pi, window, frac, IS_func,
                  accept_rate, up_rate, pi_hat, runtime, iterations, 
                  T, K, J, notes))
        connection.commit()

create_table()
