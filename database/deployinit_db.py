# database/init_db_sqlite.py
import pandas as pd
import sqlite3
from pathlib import Path
import time
from datetime import datetime
import json

class SQLiteBulkLoader:
    def __init__(self, db_path="procurement.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish connection to SQLite"""
        print("🔗 Connecting to SQLite...")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.conn.execute("PRAGMA foreign_keys = ON")
        print(f"✅ Connected to SQLite database: {self.db_path}")
        return self.conn
    
    def disconnect(self):
        """Close connection"""
        if self.conn:
            self.conn.commit()
            self.conn.close()
        print("✅ Connection closed")
    
    def drop_tables(self):
        """Drop existing tables safely"""
        print("\n🗑️  Dropping existing tables...")
        tables = [
            'procurement_anomalies',
            'procurements', 
            'vendors', 
            'agencies', 
            'world_bank_indicators'
        ]
        
        for table in tables:
            try:
                self.cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"   ✅ Dropped: {table}")
            except Exception as e:
                print(f"   ℹ️  {table} didn't exist: {e}")
        
        self.conn.commit()
    
    def create_tables(self):
        """Create optimized tables"""
        print("\n🛠️  Creating tables...")
        
        # Agencies - small table
        self.cursor.execute("""
            CREATE TABLE agencies (
                agency_code TEXT PRIMARY KEY,
                agency_name TEXT,
                agency_type TEXT
            )
        """)
        print("   ✅ Created: agencies")
        
        # Vendors - medium/large table
        self.cursor.execute("""
            CREATE TABLE vendors (
                vendor_code TEXT PRIMARY KEY,
                vendor_name TEXT,
                vendor_country TEXT,
                vendor_type TEXT
            )
        """)
        print("   ✅ Created: vendors")
        
        # World Bank indicators - small table
        self.cursor.execute("""
            CREATE TABLE world_bank_indicators (
                indicator_year INTEGER PRIMARY KEY,
                gdp_growth REAL,
                inflation_rate REAL,
                government_expenditure REAL
            )
        """)
        print("   ✅ Created: world_bank_indicators")
        
        # Procurements - MAIN TABLE
        self.cursor.execute("""
            CREATE TABLE procurements (
                procurement_id INTEGER PRIMARY KEY,
                procurement_year INTEGER,
                agency_code TEXT,
                agency_name TEXT,
                project_name TEXT,
                procurement_category TEXT,
                estimated_cost REAL,
                actual_cost REAL,
                procurement_method TEXT,
                vendor_code TEXT,
                vendor_name TEXT,
                contract_start_date DATE,
                contract_end_date DATE,
                procurement_status TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("   ✅ Created: procurements")
        
        self.conn.commit()
        print("   ✅ All tables created successfully!")
    
    def create_indexes(self):
        """Create indexes for fast queries"""
        print("\n⚡ Creating performance indexes...")
        
        indexes = [
            ("idx_proc_year", "procurements(procurement_year)"),
            ("idx_proc_agency_code", "procurements(agency_code)"),
            ("idx_proc_vendor_code", "procurements(vendor_code)"),
            ("idx_proc_method", "procurements(procurement_method)"),
            ("idx_proc_status", "procurements(procurement_status)"),
            ("idx_proc_dates", "procurements(contract_start_date, contract_end_date)"),
            ("idx_proc_cost", "procurements(estimated_cost, actual_cost)"),
            ("idx_agency_type", "agencies(agency_type)"),
            ("idx_vendor_country", "vendors(vendor_country)"),
            ("idx_wb_year", "world_bank_indicators(indicator_year)")
        ]
        
        created = 0
        for idx_name, idx_columns in indexes:
            try:
                self.cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_columns}")
                print(f"   ✅ Index: {idx_name}")
                created += 1
            except Exception as e:
                print(f"   ⚠️  Index {idx_name} skipped: {e}")
        
        self.conn.commit()
        print(f"   📊 Created {created}/{len(indexes)} indexes")
    
    def insert_agencies(self, agencies_df):
        """Insert agencies data"""
        print(f"\n📥 Inserting agencies ({len(agencies_df):,} rows)...")
        start_time = time.time()
        
        # Clean data
        agencies_df = agencies_df.copy()
        agencies_df['agency_code'] = agencies_df['agency_code'].astype(str).str.strip()
        agencies_df = agencies_df.dropna(subset=['agency_code'])
        agencies_df = agencies_df.drop_duplicates(subset=['agency_code'])
        
        # Insert
        agencies_df.to_sql('agencies', self.conn, if_exists='append', index=False)
        
        elapsed = time.time() - start_time
        inserted = len(agencies_df)
        print(f"   ✅ Inserted {inserted:,} agencies in {elapsed:.2f} seconds")
        
        return inserted
    
    def insert_vendors(self, vendors_df):
        """Insert vendors data"""
        print(f"\n📥 Inserting vendors ({len(vendors_df):,} rows)...")
        start_time = time.time()
        
        # Clean data
        vendors_df = vendors_df.copy()
        vendors_df['vendor_code'] = vendors_df['vendor_code'].astype(str).str.strip()
        vendors_df = vendors_df.dropna(subset=['vendor_code'])
        vendors_df = vendors_df.drop_duplicates(subset=['vendor_code'])
        
        # Insert
        vendors_df.to_sql('vendors', self.conn, if_exists='append', index=False)
        
        elapsed = time.time() - start_time
        inserted = len(vendors_df)
        print(f"   ✅ Inserted {inserted:,} vendors in {elapsed:.2f} seconds")
        
        return inserted
    
    def insert_world_bank(self, world_bank_df):
        """Insert World Bank data"""
        print(f"\n📥 Inserting World Bank indicators ({len(world_bank_df):,} rows)...")
        start_time = time.time()
        
        # Clean data
        world_bank_df = world_bank_df.copy()
        world_bank_df = world_bank_df.dropna(subset=['indicator_year'])
        
        # Insert
        world_bank_df.to_sql('world_bank_indicators', self.conn, if_exists='append', index=False)
        
        elapsed = time.time() - start_time
        inserted = len(world_bank_df)
        print(f"   ✅ Inserted {inserted:,} rows in {elapsed:.2f} seconds")
        
        return inserted
    
    def insert_procurements(self, procurements_df):
        """Insert procurements data"""
        total_rows = len(procurements_df)
        print(f"\n📥 Inserting procurements ({total_rows:,} rows)...")
        start_time = time.time()
        
        # Clean data
        df_clean = procurements_df.copy()
        
        # Convert numeric columns
        df_clean['procurement_id'] = pd.to_numeric(df_clean['procurement_id'], errors='coerce')
        df_clean = df_clean[df_clean['procurement_id'].notna()]
        df_clean['procurement_id'] = df_clean['procurement_id'].astype('int64')
        
        df_clean['procurement_year'] = pd.to_numeric(df_clean['procurement_year'], errors='coerce').fillna(2023).astype('int64')
        df_clean['estimated_cost'] = pd.to_numeric(df_clean['estimated_cost'], errors='coerce').fillna(0.0)
        df_clean['actual_cost'] = pd.to_numeric(df_clean['actual_cost'], errors='coerce')
        
        # Clean strings
        string_cols = ['agency_code', 'agency_name', 'project_name', 'procurement_category',
                      'procurement_method', 'vendor_code', 'vendor_name', 'procurement_status']
        
        for col in string_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).fillna('UNKNOWN').str.strip()
        
        # Clean dates
        df_clean['contract_start_date'] = pd.to_datetime(df_clean['contract_start_date'], errors='coerce')
        df_clean['contract_end_date'] = pd.to_datetime(df_clean['contract_end_date'], errors='coerce')
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['procurement_id'])
        
        print(f"   ✅ Cleaned data: {len(df_clean):,} valid rows")
        
        # Insert in chunks for better performance
        chunk_size = 1000
        successful_rows = 0
        
        for i in range(0, len(df_clean), chunk_size):
            chunk = df_clean.iloc[i:i+chunk_size]
            chunk.to_sql('procurements', self.conn, if_exists='append', index=False, method='multi')
            successful_rows += len(chunk)
            
            progress = min(100, (i + len(chunk)) / len(df_clean) * 100)
            print(f"   📊 Progress: {progress:.1f}% ({successful_rows:,}/{len(df_clean):,} rows)")
        
        total_time = time.time() - start_time
        
        print(f"\n✅ LOAD COMPLETE:")
        print(f"   📈 Successful: {successful_rows:,} out of {total_rows:,} rows")
        print(f"   ⏱️  Time: {total_time:.2f} seconds")
        print(f"   🚀 Speed: {successful_rows/total_time:.0f} rows/second")
        
        return successful_rows
    
    def create_anomaly_table(self):
        """Create anomaly analysis table"""
        print("\n🔍 Creating anomaly analysis table...")
        
        try:
            # Drop if exists
            self.cursor.execute("DROP TABLE IF EXISTS procurement_anomalies")
            
            self.cursor.execute("""
                CREATE TABLE procurement_anomalies AS
                SELECT 
                    p.*,
                    CASE WHEN p.actual_cost > p.estimated_cost * 1.1 THEN 1 ELSE 0 END 
                        AS high_cost_overrun,
                    CASE WHEN p.estimated_cost > 1000000 
                         AND p.procurement_method IN ('Direct', 'Limited') THEN 1 ELSE 0 END 
                        AS large_direct_contract,
                    CASE WHEN (julianday(p.contract_end_date) - julianday(p.contract_start_date)) < 30 THEN 1 ELSE 0 END 
                        AS short_duration_flag,
                    (CASE WHEN p.actual_cost > p.estimated_cost * 1.1 THEN 1 ELSE 0 END +
                     CASE WHEN p.estimated_cost > 1000000 
                          AND p.procurement_method IN ('Direct', 'Limited') THEN 1 ELSE 0 END +
                     CASE WHEN (julianday(p.contract_end_date) - julianday(p.contract_start_date)) < 30 THEN 1 ELSE 0 END) 
                        AS total_risk_score
                FROM procurements p
            """)
            
            # Add indexes
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_risk ON procurement_anomalies(total_risk_score)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_overrun ON procurement_anomalies(high_cost_overrun)")
            
            self.conn.commit()
            print("   ✅ Anomaly table created with indexes")
            
        except Exception as e:
            print(f"   ⚠️  Anomaly table skipped: {e}")
    
    def verify_data(self):
        """Verify all data was loaded correctly"""
        print("\n🔍 Verifying data integrity...")
        
        tables = ['agencies', 'vendors', 'world_bank_indicators', 'procurements']
        
        for table in tables:
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                print(f"   {table.upper():25} {count:>12,} records")
            except:
                print(f"   {table.upper():25} {'N/A':>12}")
        
        print("\n✅ Verification complete")

def main():
    """Main function"""
    print("=" * 80)
    print("🚀 SQLite Procurement Data Loader")
    print("=" * 80)
    
    # Configuration - using relative paths
    DATA_PATH = Path("data")
    DB_PATH = "procurement.db"
    
    # Check data files
    print("\n📁 Checking data files...")
    required_files = {
        "agencies.csv": "agencies",
        "vendors.csv": "vendors", 
        "world_bank_indicators.csv": "world_bank",
        "procurements.csv": "procurements"
    }
    
    for filename, description in required_files.items():
        filepath = DATA_PATH / filename
        if filepath.exists():
            row_count = sum(1 for line in open(filepath, encoding='utf-8')) - 1
            print(f"   ✅ {filename}: {row_count:,} {description} rows")
        else:
            print(f"   ❌ Missing: {filename}")
            print(f"\n⚠️  Please add {filename} to {DATA_PATH}/")
            return
    
    # Initialize loader
    loader = SQLiteBulkLoader(DB_PATH)
    
    try:
        # 1. Connect
        loader.connect()
        
        # 2. Drop old tables
        loader.drop_tables()
        
        # 3. Create new tables
        loader.create_tables()
        
        # 4. Load data
        print("\n" + "=" * 80)
        print("📥 LOADING DATA...")
        print("=" * 80)
        
        total_start = time.time()
        
        # Load agencies
        agencies_df = pd.read_csv(DATA_PATH / "agencies.csv", dtype=str)
        agencies_count = loader.insert_agencies(agencies_df)
        
        # Load vendors
        vendors_df = pd.read_csv(DATA_PATH / "vendors.csv", dtype=str)
        vendors_count = loader.insert_vendors(vendors_df)
        
        # Load world bank
        world_bank_df = pd.read_csv(DATA_PATH / "world_bank_indicators.csv")
        world_bank_count = loader.insert_world_bank(world_bank_df)
        
        # Load procurements (main data)
        print("\n" + "-" * 80)
        print("📊 LOADING PROCUREMENTS (Main Dataset)")
        print("-" * 80)
        
        procurements_df = pd.read_csv(
            DATA_PATH / "procurements.csv",
            low_memory=False
        )
        
        procurements_count = loader.insert_procurements(procurements_df)
        
        # 5. Create indexes AFTER data is loaded
        print("\n" + "=" * 80)
        print("⚡ OPTIMIZING DATABASE...")
        print("=" * 80)
        
        loader.create_indexes()
        
        # 6. Create anomaly table
        loader.create_anomaly_table()
        
        # 7. Verify
        loader.verify_data()
        
        # 8. Final statistics
        total_time = time.time() - total_start
        total_rows = agencies_count + vendors_count + world_bank_count + procurements_count
        
        print("\n" + "=" * 80)
        print("🎉 DATA LOAD COMPLETE!")
        print("=" * 80)
        
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   ⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   📈 Total rows loaded: {total_rows:,}")
        print(f"   🚀 Average speed: {total_rows/total_time:.0f} rows/second")
        
        print(f"\n📁 DATA BREAKDOWN:")
        print(f"   • Agencies: {agencies_count:,} rows")
        print(f"   • Vendors: {vendors_count:,} rows")
        print(f"   • World Bank: {world_bank_count:,} rows")
        print(f"   • Procurements: {procurements_count:,} rows")
        
        print(f"\n💾 Database saved to: {DB_PATH}")
        print(f"   Size: {Path(DB_PATH).stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        loader.disconnect()

if __name__ == "__main__":
    main()