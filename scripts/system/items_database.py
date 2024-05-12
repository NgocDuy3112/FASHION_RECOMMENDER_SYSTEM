from __init__ import *

class ItemsDatabaseModule():
    def __init__(self, dbname="postgres", user="mac", host="localhost", password="DiDi3112!", port=5432):
        self.conn = psycopg.connect(dbname=dbname, user=user, host=host, password=password, port=port)
        self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.conn)
        self.cursor = self.conn.cursor()
        self.table_name = None
        
    def is_connected(self):
        try:
            self.cursor.execute('SELECT 1')
            print("Database connection established")
            return True
        except (Exception, psycopg.DatabaseError) as error:
            print(error)
            return False
        
    def create_index(self, distance_name='vector_l2_ops', m=64, ef_construction=64):
        self.cursor.execute(
            f"""
                CREATE INDEX ON {self.table_name} 
                USING hnsw (embedding {distance_name})
                WITH (m = {m}, ef_construction = {ef_construction})
            """)
        self.conn.commit()
        
    def drop_index(self, index_name='items_embedding_idx'):
        self.cursor.execute(f"DROP INDEX IF EXISTS {index_name};")
        self.conn.commit()
        
    def set_num_workers(self, max_parallel_maintenance_workers=4):
        self.conn.execute(f"SET max_parallel_maintenance_workers = {max_parallel_maintenance_workers}")
        self.conn.commit()
        
    def set_search_size(self, ef_search=40):
        self.cursor.execute(f"SET hnsw.ef_search = {ef_search};")
        self.conn.commit()
        
    def query(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def close(self):
        self.cursor.close()
        self.conn.close()
        
    def create_table(self, table_name, embedding_size=256):
        self.table_name = table_name
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY, 
                image TEXT, 
                category TEXT,
                embedding vector({embedding_size})
            );
        """
        self.cursor.execute(query)
        self.conn.commit()
        
    def get_table_name(self):
        query = f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
            AND table_type='BASE TABLE';
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()[0][0]
        
    def insert_value(self, image, category, embedding):
        query = f"""
            INSERT INTO {self.table_name} (image, category, embedding)
            VALUES (%s, %s, %s);
        """
        self.cursor.execute(query, (image, category, embedding))
        self.conn.commit()
        
    def drop_table(self, table_name):
        query = f"""
            DROP TABLE IF EXISTS {table_name};
        """
        self.cursor.execute(query)
        self.conn.commit()
        
if __name__ == "__main__":
    db = ItemsDatabaseModule()
    db.is_connected()
    print(db.get_table_name())
    db.close()