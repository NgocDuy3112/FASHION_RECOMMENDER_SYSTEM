from __init__ import *

class ItemsDatabaseModule():
    def __init__(self, dbname="postgres", user="mac", host="localhost", password="DiDi3112!", port=5432):
        try:
            self.conn = psycopg.connect(dbname=dbname, user=user, host=host, password=password, port=port)
            self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(self.conn)
            self.cursor = self.conn.cursor()
            print("Database connection established successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error connecting to database: {error}")

    def is_connected(self):
        try:
            self.cursor.execute('SELECT 1')
            print("Database connection is active")
            return True
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Database connection error: {error}")
            return False

    def create_index(self, distance_name='vector_l2_ops', m=64, ef_construction=64):
        try:
            self.cursor.execute(
                f"""
                CREATE INDEX ON {self.table_name} 
                USING hnsw (embedding {distance_name})
                WITH (m = {m}, ef_construction = {ef_construction})
                """)
            self.conn.commit()
            print("Index created successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error creating index: {error}")

    def drop_index(self, index_name='items_embedding_idx'):
        try:
            self.cursor.execute(f"DROP INDEX IF EXISTS {index_name};")
            self.conn.commit()
            print("Index dropped successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error dropping index: {error}")

    def set_num_workers(self, max_parallel_maintenance_workers=4):
        try:
            self.conn.execute(f"SET max_parallel_maintenance_workers = {max_parallel_maintenance_workers}")
            self.conn.commit()
            print("Number of workers set successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error setting number of workers: {error}")

    def set_search_size(self, ef_search=40):
        try:
            self.cursor.execute(f"SET hnsw.ef_search = {ef_search};")
            self.conn.commit()
            print("Search size set successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error setting search size: {error}")

    def query(self, query):
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error executing query: {error}")
            return None

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
            print("Database connection closed successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error closing database connection: {error}")

    def create_table(self, table_name, embedding_size=512):
        self.table_name = table_name
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY, 
                image TEXT, 
                category TEXT,
                embedding vector({embedding_size})
            );
        """
        try:
            self.cursor.execute(query)
            self.conn.commit()
            print("Table created successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error creating table: {error}")

    def get_table_name(self):
        query = f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
            AND table_type='BASE TABLE';
        """
        try:
            self.cursor.execute(query)
            table_name = self.cursor.fetchall()[0][0]
            print("Table name retrieved successfully")
            return table_name
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error getting table name: {error}")
            return None

    def insert_value(self, table_name, image, category, embedding):
        query = f"""
            INSERT INTO {table_name} (image, category, embedding)
            VALUES (%s, %s, %s);
        """
        try:
            self.cursor.execute(query, (image, category, embedding))
            self.conn.commit()
            print("Value inserted successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error inserting value: {error}")

    def drop_table(self, table_name):
        query = f"""
            DROP TABLE IF EXISTS {table_name};
        """
        try:
            self.cursor.execute(query)
            self.conn.commit()
            print("Table dropped successfully")
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error dropping table: {error}")
