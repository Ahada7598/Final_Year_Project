# dashboard/app.py
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime, timedelta
import os
import hashlib
import json
from pathlib import Path

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="Procurement Anomaly Detection",
    page_icon="🏛️",
    layout="wide"
)

# ==================== DATABASE CONNECTION ====================
@st.cache_resource
def get_connection():
    """Get SQLite database connection"""
    db_path = Path("procurement.db")
    if not db_path.exists():
        # Create database if it doesn't exist
        from database.init_db_sqlite import SQLiteBulkLoader
        loader = SQLiteBulkLoader(str(db_path))
        loader.connect()
        loader.disconnect()
    
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from SQLite"""
    try:
        conn = get_connection()
        
        # Load data with SQL queries
        procurements = pd.read_sql_query("""
            SELECT procurement_id, procurement_year, agency_code, agency_name, 
                   project_name, procurement_category, estimated_cost, actual_cost,
                   procurement_method, vendor_code, vendor_name,
                   contract_start_date, contract_end_date,
                   procurement_status
            FROM procurements
        """, conn)
        
        agencies = pd.read_sql_query("SELECT * FROM agencies", conn)
        vendors = pd.read_sql_query("SELECT * FROM vendors", conn)
        world_bank = pd.read_sql_query("SELECT * FROM world_bank_indicators", conn)
        
        # Convert dates and numeric columns
        date_cols = ['contract_start_date', 'contract_end_date']
        for col in date_cols:
            if col in procurements.columns:
                procurements[col] = pd.to_datetime(procurements[col], errors='coerce')
        
        for col in ['estimated_cost', 'actual_cost']:
            if col in procurements.columns:
                procurements[col] = pd.to_numeric(procurements[col], errors='coerce')
        
        return procurements, agencies, vendors, world_bank
        
    except Exception as e:
        st.error(f"Database error: {e}")
        # Return empty DataFrames
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ==================== AUTHENTICATION SYSTEM ====================
class AuthenticationSystem:
    """Simple authentication system for 3 roles"""
    
    USERS = {
        "viewer": {
            "password_hash": hashlib.sha256("view123".encode()).hexdigest(),
            "role": "viewer",
            "name": "Analyst Viewer"
        },
        "admin": {
            "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
            "role": "admin",
            "name": "System Administrator"
        },
        "buyer": {
            "password_hash": hashlib.sha256("buy123".encode()).hexdigest(),
            "role": "buyer",
            "name": "Vendor Buyer"
        }
    }
    
    @staticmethod
    def login(username, password):
        """Verify user credentials"""
        if username in AuthenticationSystem.USERS:
            stored_hash = AuthenticationSystem.USERS[username]["password_hash"]
            input_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if stored_hash == input_hash:
                return {
                    "authenticated": True,
                    "username": username,
                    "role": AuthenticationSystem.USERS[username]["role"],
                    "name": AuthenticationSystem.USERS[username]["name"]
                }
        return {"authenticated": False}
    
    @staticmethod
    def logout():
        """Clear session state"""
        for key in ['authenticated', 'username', 'role', 'name']:
            if key in st.session_state:
                del st.session_state[key]

# ==================== ANOMALY DETECTION FUNCTIONS ====================
def calculate_anomalies(procurements):
    """Calculate anomaly flags and risk scores"""
    df = procurements.copy()
    
    # Cost overrun percentage
    df['cost_overrun_%'] = 0.0
    mask = (df['estimated_cost'] > 0) & df['actual_cost'].notna()
    df.loc[mask, 'cost_overrun_%'] = (
        (df['actual_cost'] - df['estimated_cost']) / df['estimated_cost'] * 100
    )
    
    # Anomaly flags
    df['high_overrun_flag'] = df['cost_overrun_%'] > 10
    df['large_direct_flag'] = (
        (df['estimated_cost'] > 1000000) & 
        df['procurement_method'].isin(['Direct', 'Limited'])
    )
    
    valid_dates = df['contract_start_date'].notna() & df['contract_end_date'].notna()
    df['contract_duration'] = 0
    df.loc[valid_dates, 'contract_duration'] = (
        df['contract_end_date'] - df['contract_start_date']
    ).dt.days
    df['short_duration_flag'] = df['contract_duration'] < 30
    
    # Risk score
    df['risk_score'] = (
        df['high_overrun_flag'].astype(int) +
        df['large_direct_flag'].astype(int) +
        df['short_duration_flag'].astype(int)
    )
    
    return df

# ==================== BIDDING SYSTEM FUNCTIONS ====================
class BiddingSystem:
    """Simple bidding system"""
    
    @staticmethod
    def load_bids():
        """Load bids from database"""
        try:
            conn = get_connection()
            bids_df = pd.read_sql_query("SELECT * FROM bids ORDER BY bid_date DESC", conn)
            return bids_df.to_dict('records')
        except:
            return []
    
    @staticmethod
    def submit_bid(procurement_id, vendor_code, vendor_name, bid_amount):
        """Submit a new bid to database"""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Create bids table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bids (
                    bid_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    procurement_id INTEGER,
                    vendor_code TEXT,
                    vendor_name TEXT,
                    bid_amount REAL,
                    bid_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'Submitted'
                )
            """)
            
            # Insert bid
            cursor.execute("""
                INSERT INTO bids (procurement_id, vendor_code, vendor_name, bid_amount)
                VALUES (?, ?, ?, ?)
            """, (procurement_id, vendor_code, vendor_name, bid_amount))
            
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Bid submission error: {e}")
            return False

# ==================== LOGIN PAGE ====================
def show_login_page():
    """Display login page with 3 role options"""
    
    st.title("🏛️ Procurement Anomaly Detection System")
    st.markdown("### **Multi-User Portal**")
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 30px; border-radius: 10px;'>
        <h3>Select Your Role to Continue</h3>
        <p>Choose your role to access the appropriate dashboard features</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""<div style='text-align: center; padding: 20px; border: 2px solid #1E90FF; border-radius: 10px;'>
            <h4>👁️ VIEWER</h4><p>Read-only access</p><p>• View analytics</p><p>• ML predictions</p><p>• No editing</p>
        </div>""", unsafe_allow_html=True)
        
        with st.expander("Login as Viewer", expanded=False):
            username = st.text_input("Username", key="viewer_user")
            password = st.text_input("Password", type="password", key="viewer_pass")
            if st.button("Login as Viewer", type="primary"):
                result = AuthenticationSystem.login(username, password)
                if result["authenticated"] and result["role"] == "viewer":
                    for key, value in result.items():
                        st.session_state[key] = value
                    st.rerun()
                else:
                    st.error("Invalid viewer credentials")
    
    with col2:
        st.markdown("""<div style='text-align: center; padding: 20px; border: 2px solid #32CD32; border-radius: 10px;'>
            <h4>⚙️ ADMIN</h4><p>Full system control</p><p>• Edit all data</p><p>• Manage users</p><p>• System settings</p>
        </div>""", unsafe_allow_html=True)
        
        with st.expander("Login as Admin", expanded=False):
            username = st.text_input("Username", key="admin_user")
            password = st.text_input("Password", type="password", key="admin_pass")
            if st.button("Login as Admin", type="primary"):
                result = AuthenticationSystem.login(username, password)
                if result["authenticated"] and result["role"] == "admin":
                    for key, value in result.items():
                        st.session_state[key] = value
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")
    
    with col3:
        st.markdown("""<div style='text-align: center; padding: 20px; border: 2px solid #FF8C00; border-radius: 10px;'>
            <h4>💰 BUYER</h4><p>Bidding access</p><p>• View open projects</p><p>• Submit bids</p><p>• Track bids</p>
        </div>""", unsafe_allow_html=True)
        
        with st.expander("Login as Buyer", expanded=False):
            username = st.text_input("Username", key="buyer_user")
            password = st.text_input("Password", type="password", key="buyer_pass")
            if st.button("Login as Buyer", type="primary"):
                result = AuthenticationSystem.login(username, password)
                if result["authenticated"] and result["role"] == "buyer":
                    for key, value in result.items():
                        st.session_state[key] = value
                    st.rerun()
                else:
                    st.error("Invalid buyer credentials")
    
    st.markdown("---")
    st.markdown("""
    **Demo Credentials:**
    - **Viewer**: username=`viewer`, password=`view123`
    - **Admin**: username=`admin`, password=`admin123`
    - **Buyer**: username=`buyer`, password=`buy123`
    """)

# ==================== VIEWER DASHBOARD ====================
def show_viewer_dashboard():
    """Dashboard for Viewer role (read-only)"""
    
    # Load data
    procurements, agencies, vendors, world_bank = load_data()
    
    if procurements.empty:
        st.error("No data available. Please initialize the database.")
        if st.button("Initialize Database"):
            from database.init_db_sqlite import main as init_db
            init_db()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        return
    
    # Calculate anomalies
    procurements = calculate_anomalies(procurements)
    
    # Header with logout
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.title("📊 Procurement Analytics Dashboard")
        st.markdown(f"**Welcome, {st.session_state.name} (Viewer)**")
    with col3:
        if st.button("🚪 Logout", type="secondary"):
            AuthenticationSystem.logout()
            st.rerun()
    
    st.markdown("---")
    
    # Executive Summary
    st.header("📈 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Contracts", len(procurements))
    with col2:
        high_risk = len(procurements[procurements['risk_score'] >= 2])
        st.metric("High Risk Contracts", high_risk)
    with col3:
        cost_overruns = procurements['high_overrun_flag'].sum()
        st.metric("Cost Overruns >10%", int(cost_overruns))
    with col4:
        large_direct = procurements['large_direct_flag'].sum()
        st.metric("Large Direct Contracts", int(large_direct))
    
    st.markdown("---")
    
    # High Risk Contracts
    st.header("🚨 High Risk Contracts")
    high_risk_df = procurements[procurements['risk_score'] >= 2].sort_values('risk_score', ascending=False)
    
    if not high_risk_df.empty:
        st.dataframe(
            high_risk_df[['procurement_id', 'agency_name', 'vendor_name', 
                         'estimated_cost', 'actual_cost', 'cost_overrun_%',
                         'procurement_method', 'risk_score']].head(10),
            use_container_width=True
        )
    else:
        st.info("✅ No high-risk contracts detected")
    
    st.markdown("---")
    
    # ML Anomaly Detection
    st.header("🤖 Machine Learning Anomaly Detection")
    
    with st.expander("Run ML Analysis", expanded=False):
        st.info("""
        **Isolation Forest Algorithm** detects anomalies by isolating outliers in the data.
        Features used: Cost, Duration, Vendor Frequency
        """)
        
        if st.button("🔍 Run ML Detection", type="secondary"):
            try:
                from sklearn.ensemble import IsolationForest
                import numpy as np
                
                with st.spinner("Running ML analysis..."):
                    # Prepare features
                    features = pd.DataFrame()
                    features['log_cost'] = np.log1p(procurements['estimated_cost'].fillna(0))
                    
                    # Duration feature
                    features['duration'] = procurements['contract_duration'].fillna(0)
                    
                    # Vendor frequency
                    if 'vendor_code' in procurements.columns:
                        vendor_counts = procurements['vendor_code'].value_counts()
                        features['vendor_freq'] = procurements['vendor_code'].map(vendor_counts).fillna(1)
                    else:
                        features['vendor_freq'] = 1
                    
                    features = features.fillna(0)
                    
                    # Train Isolation Forest model
                    model = IsolationForest(
                        contamination=0.15,
                        random_state=42,
                        n_estimators=100
                    )
                    
                    ml_predictions = model.fit_predict(features)
                    procurements['ml_anomaly'] = (ml_predictions == -1)
                    procurements['ml_confidence'] = model.decision_function(features) * -1
                    
                    ml_anomalies = procurements[procurements['ml_anomaly'] == True]
                    
                    st.success(f"✅ ML detected **{len(ml_anomalies)}** anomalous contracts!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Contracts", len(procurements))
                    with col2:
                        st.metric("ML Anomalies", len(ml_anomalies))
                    with col3:
                        anomaly_percent = (len(ml_anomalies) / len(procurements)) * 100
                        st.metric("Anomaly %", f"{anomaly_percent:.1f}%")
                    
                    if not ml_anomalies.empty:
                        st.subheader(f"ML-Detected Anomalies ({len(ml_anomalies)} records)")
                        
                        ml_anomalies_sorted = ml_anomalies.sort_values('ml_confidence', ascending=False)
                        
                        st.dataframe(
                            ml_anomalies_sorted[[
                                'procurement_id', 'agency_name', 'vendor_name',
                                'estimated_cost', 'contract_duration', 'ml_confidence'
                            ]],
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download button
                        csv_data = ml_anomalies_sorted.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download ML Anomalies (CSV)",
                            data=csv_data,
                            file_name=f"ml_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="secondary"
                        )
                    else:
                        st.info("✅ No anomalies detected by ML algorithm")
                        
            except Exception as e:
                st.error(f"ML Error: {e}")
                st.info("To install required package: `pip install scikit-learn`")
    
    # Visualizations
    st.header("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        if not procurements.empty:
            fig1 = px.histogram(
                procurements[procurements['estimated_cost'] <= procurements['estimated_cost'].quantile(0.95)],
                x='estimated_cost',
                title='Contract Value Distribution',
                nbins=15,
                color_discrete_sequence=['#3366CC']
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if not procurements.empty:
            method_counts = procurements['procurement_method'].value_counts()
            fig2 = px.pie(
                values=method_counts.values,
                names=method_counts.index,
                title='Procurement Methods',
                hole=0.3
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Export Data
    st.markdown("---")
    st.header("📥 Export Data")
    
    if not procurements.empty:
        csv_data = procurements.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Analytics Data (CSV)",
            data=csv_data,
            file_name="procurement_analytics.csv",
            mime="text/csv"
        )

# ==================== ADMIN DASHBOARD ====================
def show_admin_dashboard():
    """Dashboard for Admin role (full control)"""
    
    # Load data
    procurements, agencies, vendors, world_bank = load_data()
    
    if procurements.empty:
        st.error("No data available.")
        return
    
    # Calculate anomalies
    procurements = calculate_anomalies(procurements)
    
    # Header
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.title("⚙️ Admin Control Panel")
        st.markdown(f"**Welcome, {st.session_state.name} (Administrator)**")
    with col3:
        if st.button("🚪 Logout", type="secondary"):
            AuthenticationSystem.logout()
            st.rerun()
    
    st.markdown("---")
    
    # Data Management Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Edit Data", "➕ Add New", "🗑️ Delete", "📊 Analytics"])
    
    with tab1:
        st.header("Edit Procurement Records")
        
        search_id = st.text_input("Search by Procurement ID:")
        
        if search_id:
            record = procurements[procurements['procurement_id'].astype(str) == search_id]
            
            if not record.empty:
                record = record.iloc[0]
                
                with st.form("edit_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        new_estimated = st.number_input("Estimated Cost", value=float(record['estimated_cost']))
                        new_actual = st.number_input("Actual Cost", value=float(record['actual_cost']) if pd.notna(record['actual_cost']) else 0.0)
                        new_method = st.selectbox(
                            "Procurement Method",
                            ["Open", "Limited", "Direct", "Single"],
                            index=["Open", "Limited", "Direct", "Single"].index(record['procurement_method']) 
                            if record['procurement_method'] in ["Open", "Limited", "Direct", "Single"] else 0
                        )
                    
                    with col2:
                        new_status = st.selectbox(
                            "Status",
                            ["Ongoing", "Completed", "Cancelled"],
                            index=["Ongoing", "Completed", "Cancelled"].index(record['procurement_status']) 
                            if record['procurement_status'] in ["Ongoing", "Completed", "Cancelled"] else 0
                        )
                        
                        if pd.notna(record['contract_start_date']):
                            new_start = st.date_input("Start Date", value=record['contract_start_date'].date())
                        else:
                            new_start = st.date_input("Start Date")
                        
                        if pd.notna(record['contract_end_date']):
                            new_end = st.date_input("End Date", value=record['contract_end_date'].date())
                        else:
                            new_end = st.date_input("End Date")
                    
                    if st.form_submit_button("💾 Save Changes", type="primary"):
                        # Update database
                        try:
                            conn = get_connection()
                            cursor = conn.cursor()
                            cursor.execute("""
                                UPDATE procurements 
                                SET estimated_cost = ?, actual_cost = ?, procurement_method = ?,
                                    procurement_status = ?, contract_start_date = ?, contract_end_date = ?
                                WHERE procurement_id = ?
                            """, (new_estimated, new_actual, new_method, new_status, 
                                  new_start.isoformat(), new_end.isoformat(), int(search_id)))
                            conn.commit()
                            st.success(f"✅ Record {search_id} updated successfully!")
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    with tab2:
        st.header("Add New Procurement Record")
        
        with st.form("add_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_id = st.number_input("Procurement ID", min_value=1, step=1)
                new_year = st.number_input("Year", min_value=2010, max_value=2024, value=2023)
                agency = st.selectbox("Agency", agencies['agency_name'].tolist())
                project = st.text_input("Project Name")
                category = st.selectbox("Category", ["Works", "Goods", "Services"])
            
            with col2:
                estimated = st.number_input("Estimated Cost ($)", min_value=0.0, value=100000.0)
                actual = st.number_input("Actual Cost ($)", min_value=0.0, value=0.0)
                method = st.selectbox("Method", ["Open", "Limited", "Direct", "Single"])
                vendor = st.selectbox("Vendor", vendors['vendor_name'].tolist()[:50])
                status = st.selectbox("Status", ["Ongoing", "Completed"])
            
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            
            if st.form_submit_button("➕ Add New Record", type="primary"):
                try:
                    conn = get_connection()
                    cursor = conn.cursor()
                    
                    # Get agency and vendor codes
                    agency_code = agencies[agencies['agency_name'] == agency]['agency_code'].iloc[0]
                    vendor_code = vendors[vendors['vendor_name'] == vendor]['vendor_code'].iloc[0]
                    
                    cursor.execute("""
                        INSERT INTO procurements 
                        (procurement_id, procurement_year, agency_code, agency_name, 
                         project_name, procurement_category, estimated_cost, actual_cost,
                         procurement_method, vendor_code, vendor_name, 
                         contract_start_date, contract_end_date, procurement_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (int(new_id), int(new_year), agency_code, agency, project, category,
                          estimated, actual, method, vendor_code, vendor,
                          start_date.isoformat(), end_date.isoformat(), status))
                    
                    conn.commit()
                    st.success("✅ New record added successfully!")
                    st.cache_data.clear()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab3:
        st.header("Delete Records")
        st.warning("⚠️ **CAUTION**: This action cannot be undone!")
        
        delete_id = st.text_input("Enter Procurement ID to delete:")
        
        if delete_id:
            record_exists = delete_id in procurements['procurement_id'].astype(str).values
            
            if record_exists:
                st.error(f"Record {delete_id} will be permanently deleted!")
                
                if st.button("🗑️ Confirm Delete", type="primary"):
                    try:
                        conn = get_connection()
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM procurements WHERE procurement_id = ?", (int(delete_id),))
                        conn.commit()
                        st.success(f"✅ Record {delete_id} deleted successfully!")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning(f"No record found with ID: {delete_id}")
    
    with tab4:
        # Show analytics
        show_viewer_dashboard_content(procurements)

# ==================== BUYER DASHBOARD ====================
def show_buyer_dashboard():
    """Dashboard for Buyer role (bidding)"""
    
    # Load data
    procurements, agencies, vendors, world_bank = load_data()
    
    if procurements.empty:
        st.error("No data available.")
        return
    
    # Header
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.title("💰 Buyer Portal")
        st.markdown(f"**Welcome, {st.session_state.name} (Buyer)**")
    with col3:
        if st.button("🚪 Logout", type="secondary"):
            AuthenticationSystem.logout()
            st.rerun()
    
    st.markdown("---")
    
    # Tab interface
    tab1, tab2, tab3 = st.tabs(["🏆 Open Projects", "📝 Submit Bid", "📋 My Bids"])
    
    with tab1:
        st.header("🏆 Open Procurement Projects")
        
        open_projects = procurements[procurements['procurement_status'] == 'Ongoing']
        
        if not open_projects.empty:
            st.success(f"Found {len(open_projects)} open projects for bidding")
            
            for idx, project in open_projects.head(10).iterrows():
                with st.expander(f"📋 {project['project_name']} (ID: {project['procurement_id']})", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Agency:** {project['agency_name']}")
                        st.markdown(f"**Category:** {project['procurement_category']}")
                        st.markdown(f"**Method:** {project['procurement_method']}")
                    
                    with col2:
                        st.markdown(f"**Estimated Cost:** ${project['estimated_cost']:,.2f}")
                        if pd.notna(project['contract_start_date']):
                            st.markdown(f"**Start Date:** {project['contract_start_date'].strftime('%Y-%m-%d')}")
                        if pd.notna(project['contract_end_date']):
                            st.markdown(f"**End Date:** {project['contract_end_date'].strftime('%Y-%m-%d')}")
                    
                    if st.button(f"💰 Bid on this Project", key=f"bid_btn_{project['procurement_id']}"):
                        st.session_state.selected_project = project['procurement_id']
                        st.rerun()
        else:
            st.info("No open projects available for bidding")
    
    with tab2:
        st.header("📝 Submit New Bid")
        
        if 'selected_project' in st.session_state:
            selected_id = st.session_state.selected_project
            project = procurements[procurements['procurement_id'] == selected_id]
            
            if not project.empty:
                project = project.iloc[0]
                st.success(f"Bidding on: {project['project_name']}")
                
                with st.form("bid_form"):
                    st.markdown(f"**Project:** {project['project_name']}")
                    st.markdown(f"**Agency:** {project['agency_name']}")
                    st.markdown(f"**Estimated Budget:** ${project['estimated_cost']:,.2f}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        vendor_name = st.text_input("Your Company Name", value=st.session_state.name)
                        vendor_email = st.text_input("Contact Email")
                    
                    with col2:
                        bid_amount = st.number_input(
                            "Your Bid Amount ($)",
                            min_value=0.0,
                            value=float(project['estimated_cost']) * 0.9
                        )
                        bid_deadline = st.date_input("Proposed Completion Date")
                    
                    bid_details = st.text_area("Bid Proposal Details", height=100)
                    
                    if st.form_submit_button("📤 Submit Bid", type="primary"):
                        if BiddingSystem.submit_bid(
                            selected_id,
                            "V999",
                            vendor_name,
                            bid_amount
                        ):
                            st.success("✅ Bid submitted successfully!")
                            st.balloons()
                            del st.session_state.selected_project
                        else:
                            st.error("Failed to submit bid")
            else:
                st.warning("Selected project not found")
                del st.session_state.selected_project
        else:
            st.info("Select a project from 'Open Projects' tab to submit a bid")
    
    with tab3:
        st.header("📋 My Submitted Bids")
        
        bids = BiddingSystem.load_bids()
        
        if bids:
            st.success(f"You have submitted {len(bids)} bids")
            
            for bid in bids:
                with st.expander(f"Bid #{bid['bid_id']} - Project {bid['procurement_id']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Bid Amount:** ${bid['bid_amount']:,.2f}")
                        st.markdown(f"**Submitted:** {bid['bid_date']}")
                        st.markdown(f"**Status:** {bid['status']}")
                    
                    with col2:
                        project = procurements[procurements['procurement_id'] == bid['procurement_id']]
                        if not project.empty:
                            project = project.iloc[0]
                            st.markdown(f"**Project:** {project['project_name']}")
                            st.markdown(f"**Agency:** {project['agency_name']}")
        else:
            st.info("You haven't submitted any bids yet")

# ==================== VIEWER CONTENT (reusable) ====================
def show_viewer_dashboard_content(procurements):
    """Shared viewer content for admin dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Contracts", len(procurements))
    with col2:
        high_risk = len(procurements[procurements['risk_score'] >= 2])
        st.metric("High Risk Contracts", high_risk)
    with col3:
        cost_overruns = procurements['high_overrun_flag'].sum()
        st.metric("Cost Overruns >10%", int(cost_overruns))
    with col4:
        large_direct = procurements['large_direct_flag'].sum()
        st.metric("Large Direct Contracts", int(large_direct))

# ==================== MAIN APPLICATION FLOW ====================
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Check authentication status
    if not st.session_state.authenticated:
        show_login_page()
    else:
        # Show appropriate dashboard based on role
        role = st.session_state.get('role', 'viewer')
        
        if role == 'viewer':
            show_viewer_dashboard()
        elif role == 'admin':
            show_admin_dashboard()
        elif role == 'buyer':
            show_buyer_dashboard()
        else:
            st.error("Invalid role detected")
            AuthenticationSystem.logout()
            st.rerun()

# Run the main application
if __name__ == "__main__":
    main()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 10px; border-radius: 10px;">
        <h4 style="color: #1E3A8A;">Procurement Anomaly Detection System</h4>
        <p><b>Final Year Project | Government Contract Monitoring & Fraud Detection</b></p>
        <p>🔍 Detection Rules: Cost overruns (>10%) • Large direct contracts • Short durations (<30 days)</p>
        <p>⏱️ Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """, unsafe_allow_html=True)