# Database Configuration Components
import streamlit as st
from ..utils.api_client import APIClient
from ..utils.session_state import set_file_info
from ..utils.styling import render_info_card

def render_database_config(mode: str = "fast"):
    """Render database configuration component"""
    
    st.markdown("#### 🗄️ Database Configuration")
    
    # Database type selection
    db_type = st.selectbox(
        "Database Type",
        ["mysql", "postgresql"],
        key=f"{mode}_db_type",
        help="Select the type of database to connect to"
    )
    
    # Connection parameters
    col1, col2 = st.columns(2)
    
    with col1:
        host = st.text_input(
            "Host",
            value="localhost",
            key=f"{mode}_db_host",
            help="Database server hostname or IP address"
        )
        
        port = st.number_input(
            "Port",
            value=3306 if db_type == "mysql" else 5432,
            min_value=1,
            max_value=65535,
            key=f"{mode}_db_port",
            help="Database server port"
        )
        
        username = st.text_input(
            "Username",
            key=f"{mode}_db_username",
            help="Database username"
        )
    
    with col2:
        password = st.text_input(
            "Password",
            type="password",
            key=f"{mode}_db_password",
            help="Database password"
        )
        
        database = st.text_input(
            "Database Name",
            key=f"{mode}_db_database",
            help="Name of the database to connect to"
        )
        
        table_name = st.text_input(
            "Table Name",
            key=f"{mode}_db_table",
            help="Name of the table to import"
        )
    
    # Connection test
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔌 Test Connection", key=f"{mode}_test_connection"):
            test_database_connection(db_type, host, port, username, password, database, mode)
    
    with col2:
        if st.button("📋 List Tables", key=f"{mode}_list_tables"):
            list_database_tables(db_type, host, port, username, password, database, mode)
    
    # Table selection
    if f"{mode}_available_tables" in st.session_state:
        st.markdown("#### 📋 Available Tables")
        
        available_tables = st.session_state[f"{mode}_available_tables"]
        
        if available_tables:
            selected_table = st.selectbox(
                "Select Table",
                available_tables,
                key=f"{mode}_selected_table",
                help="Select the table to import"
            )
            
            # Update table name input
            st.session_state[f"{mode}_db_table"] = selected_table
            
            # Show table info
            if f"{mode}_table_info" in st.session_state:
                table_info = st.session_state[f"{mode}_table_info"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(render_info_card(
                        "Table Name",
                        table_info.get("table_name", "N/A")
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(render_info_card(
                        "Rows",
                        str(table_info.get("rows", 0))
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(render_info_card(
                        "Columns",
                        str(table_info.get("columns", 0))
                    ), unsafe_allow_html=True)
                
                # Column information
                if "column_names" in table_info:
                    st.markdown("#### 📝 Column Information")
                    column_df = pd.DataFrame({
                        "Column": table_info["column_names"],
                        "Index": range(len(table_info["column_names"]))
                    })
                    st.dataframe(column_df, use_container_width=True)
        else:
            st.warning("No tables found in the database")
    
    # Database import tips
    with st.expander("💡 Database Import Tips"):
        st.markdown("""
        **Connection Requirements:**
        - Ensure database server is accessible
        - Valid credentials with read permissions
        - Network connectivity to database server
        
        **Supported Databases:**
        - MySQL 5.7+
        - PostgreSQL 10+
        - SQLite (local files)
        
        **Best Practices:**
        - Use dedicated read-only user account
        - Test connection before processing
        - Consider table size and memory requirements
        - Use appropriate chunking for large tables
        
        **Large Tables:**
        - Tables > 100MB may take longer to import
        - Consider using database views for filtered data
        - Monitor system memory during import
        """)

def test_database_connection(db_type: str, host: str, port: int, username: str, password: str, database: str, mode: str):
    """Test database connection"""
    try:
        api_client = APIClient()
        
        db_config = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database
        }
        
        with st.spinner("Testing connection..."):
            result = api_client.test_database_connection(db_config)
        
        if result.get("success") and result["data"].get("connected"):
            st.success("✅ Database connection successful!")
            
            # Store connection info in session state
            st.session_state[f"{mode}_db_config"] = db_config
            
            # Set file info for database import
            set_file_info({
                "filename": f"{database}.{st.session_state.get(f'{mode}_db_table', 'table')}",
                "file_type": "database_table",
                "db_config": db_config
            })
            
        else:
            error_msg = result.get("error", "Connection failed")
            st.error(f"❌ Database connection failed: {error_msg}")
    
    except Exception as e:
        st.error(f"❌ Error testing connection: {str(e)}")

def list_database_tables(db_type: str, host: str, port: int, username: str, password: str, database: str, mode: str):
    """List database tables"""
    try:
        api_client = APIClient()
        
        db_config = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database
        }
        
        with st.spinner("Fetching table list..."):
            result = api_client.list_database_tables(db_config)
        
        if result.get("success") and result["data"].get("success"):
            tables = result["data"].get("tables", [])
            st.session_state[f"{mode}_available_tables"] = tables
            
            if tables:
                st.success(f"✅ Found {len(tables)} tables")
            else:
                st.warning("⚠️ No tables found in the database")
        else:
            error_msg = result.get("error", "Failed to list tables")
            st.error(f"❌ Error listing tables: {error_msg}")
    
    except Exception as e:
        st.error(f"❌ Error listing tables: {str(e)}")

def import_database_table(db_type: str, host: str, port: int, username: str, password: str, database: str, table_name: str, mode: str):
    """Import database table"""
    try:
        api_client = APIClient()
        
        db_config = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database,
            "table_name": table_name
        }
        
        with st.spinner(f"Importing table '{table_name}'..."):
            result = api_client.import_database_table(db_config)
        
        if result.get("success") and result["data"].get("success"):
            table_info = result["data"]
            st.session_state[f"{mode}_table_info"] = table_info
            
            st.success(f"✅ Table '{table_name}' imported successfully!")
            st.info(f"Imported {table_info.get('rows', 0)} rows with {table_info.get('columns', 0)} columns")
            
            # Set file info for database import
            set_file_info({
                "filename": f"{database}.{table_name}",
                "file_type": "database_table",
                "db_config": db_config,
                "rows": table_info.get("rows", 0),
                "columns": table_info.get("columns", 0)
            })
            
        else:
            error_msg = result.get("error", "Import failed")
            st.error(f"❌ Table import failed: {error_msg}")
    
    except Exception as e:
        st.error(f"❌ Error importing table: {str(e)}")
