"""
Database module - MongoDB only for all data storage.
All data is stored in MongoDB collections.
"""

# Import MongoDB functions
from .unified_db import (
    get_mongo_db,
    UnifiedDataStore,
)

__all__ = [
    "get_mongo_db",
    "UnifiedDataStore",
]

