import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

class GraphDB:
    _driver = None

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls._driver = GraphDatabase.driver(URI, auth=AUTH)
        return cls._driver

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()

def run_query(query, params=None):
    driver = GraphDB.get_driver()
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]