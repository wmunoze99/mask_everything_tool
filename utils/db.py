from threading import Lock
from tinydb import TinyDB
from utils.progress import ProgressBarController


class _DBMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class _DBBase(metaclass=_DBMeta):
    progress = ProgressBarController()

    def __init__(self):
        self.progress.new("🚀 Creating database...")
        self.db = TinyDB('./database/classes.json')
        self.progress.terminate()

    def get_db_instance(self):
        return self.db


class DB(_DBBase):
    def __init__(self):
        super().__init__()
        self.tables = {}
        self.create_table('metadata')
        if len(self.tables['metadata'].all()) == 0:
            self.insert_into_table('metadata', {
                'total_tags': 0,
                'total_images': 0
            })

    def create_table(self, name: str):
        if not name:
            raise "A name must be provide in order to create a table"

        self.tables[name] = self.db.table(name)

    def insert_into_table(self, name: str, document: dict):
        if not name or not document:
            raise "Parameters are empty impossible to add a new document"

        self.tables[name].insert(document)

    def get_all_documents_from_table(self, name):
        return self.tables[name].all()

    def query_document(self, name, query):
        return self.tables[name].search(query)