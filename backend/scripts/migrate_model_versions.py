"""
Migration script to add missing columns to model_versions table.
Adds: source, source_url, version_id columns if they don't exist.
"""

import sqlite3
from pathlib import Path
from ..config import get_settings

settings = get_settings()

def migrate_model_versions():
    """Add missing columns to model_versions table if they don't exist."""
    # Get database path
    database_url = getattr(settings, 'database_url', None) or "sqlite:///./data/forecast.db"
    
    # Extract path from SQLite URL
    if database_url.startswith("sqlite:///"):
        db_path = database_url.replace("sqlite:///", "")
        # Handle relative paths
        if not Path(db_path).is_absolute():
            db_path = Path(__file__).parent.parent.parent / db_path
    else:
        raise ValueError(f"Unsupported database URL: {database_url}")
    
    db_path = Path(db_path)
    
    if not db_path.exists():
        print(f"Database not found at {db_path}. Creating new database...")
        # Create directory if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Check what columns exist
        cursor.execute("PRAGMA table_info(model_versions)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        print(f"Existing columns: {existing_columns}")
        
        # Add missing columns
        columns_to_add = []
        
        if 'source' not in existing_columns:
            columns_to_add.append(('source', 'VARCHAR(32)'))
            print("  - Adding 'source' column")
        
        if 'source_url' not in existing_columns:
            columns_to_add.append(('source_url', 'VARCHAR(512)'))
            print("  - Adding 'source_url' column")
        
        if 'version_id' not in existing_columns:
            columns_to_add.append(('version_id', 'VARCHAR(128)'))
            print("  - Adding 'version_id' column")
        
        # SQLite doesn't support ALTER TABLE ADD COLUMN with multiple columns at once
        for column_name, column_type in columns_to_add:
            try:
                # SQLite doesn't support IF NOT EXISTS in ALTER TABLE, so we check first
                alter_sql = f"ALTER TABLE model_versions ADD COLUMN {column_name} {column_type}"
                cursor.execute(alter_sql)
                print(f"  ✓ Added column '{column_name}'")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"  - Column '{column_name}' already exists, skipping")
                else:
                    raise
        
        # Create index on source column if it doesn't exist
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_source ON model_versions(source)")
            print("  ✓ Created index on 'source' column")
        except sqlite3.OperationalError as e:
            print(f"  - Index creation: {e}")
        
        conn.commit()
        print("\n✓ Migration completed successfully!")
        
        # Verify
        cursor.execute("PRAGMA table_info(model_versions)")
        final_columns = [row[1] for row in cursor.fetchall()]
        print(f"\nFinal columns: {final_columns}")
        
    except Exception as e:
        conn.rollback()
        print(f"\n✗ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate_model_versions()

