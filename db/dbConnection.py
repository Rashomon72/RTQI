# db.py
from pymongo import MongoClient

class MongoDB:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_db(self):
        return self.db
