#!/usr/bin/env python3
"""Quick script to view SQLite database contents."""

import sqlite3
import sys
from pathlib import Path

def view_database(db_path: str):
    """Display database tables and row counts."""
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"\n📊 Database: {db_path}")
    print("=" * 80)
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        print("No tables found.")
        return
    
    print(f"\n📋 Found {len(tables)} table(s):\n")
    
    for (table_name,) in tables:
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        print(f"  🔹 {table_name}")
        print(f"     Rows: {count}")
        print(f"     Columns: {', '.join([col[1] for col in columns])}")
        
        # Show sample data if exists
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            rows = cursor.fetchall()
            print(f"     Sample (first 3 rows):")
            for row in rows:
                print(f"       {row}")
        print()
    
    conn.close()

if __name__ == "__main__":
    # View both databases
    print("\n" + "🗄️  AI Image Detector - Database Viewer ".center(80, "="))
    
    db1 = "ai_detector.db"
    db2 = "data/simple_app.db"
    
    if len(sys.argv) > 1:
        # View specific database
        view_database(sys.argv[1])
    else:
        # View both databases
        view_database(db1)
        view_database(db2)
    
    print("=" * 80)
    print("\n💡 Usage:")
    print(f"   python view_db.py              # View both databases")
    print(f"   python view_db.py {db1}        # View specific database")
    print(f"   sqlite3 {db1}                  # Open in sqlite3 CLI")
