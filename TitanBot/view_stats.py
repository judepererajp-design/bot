import sqlite3
import pandas as pd
import os

DB_PATH = "data/titan.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}. Run the bot first.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        
        print("\n" + "="*50)
        print("📊 TITAN-X SIGNAL STATISTICS")
        print("="*50)

        # 1. Total Signals
        try:
            total = pd.read_sql_query("SELECT COUNT(*) as count FROM signals", conn)['count'][0]
            print(f"Total Signals Generated: {total}")
        except:
            print("No signals table found yet.")
            return

        # 2. Last 5 Signals
        print("\nRecent Signals:")
        recent = pd.read_sql_query(
            "SELECT symbol, pattern_name, timeframe, direction, created_at FROM signals ORDER BY id DESC LIMIT 5", 
            conn
        )
        if not recent.empty:
            print(recent.to_string(index=False))
        else:
            print("No signals yet.")

        conn.close()
    except Exception as e:
        print(f"Error reading stats: {e}")

if __name__ == "__main__":
    main()