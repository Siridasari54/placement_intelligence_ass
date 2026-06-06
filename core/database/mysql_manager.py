"""Reusable MySQL connection manager and pooling layer for Placement Intelligence Assistant."""

import logging
import mysql.connector
from mysql.connector import pooling
from config.settings import settings

logger = logging.getLogger(__name__)

class MySQLConnectionManager:
    """Singleton MySQL connection pooled manager.
    
    Handles thread-safe connection pooling, automated database bootstrapping, 
    and parameterized query execution wrappers.
    """
    _instance = None
    _pool = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MySQLConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if self._initialized:
            return
            
        self.host = settings.mysql.host
        self.port = settings.mysql.port
        self.user = settings.mysql.user
        self.password = settings.mysql.password
        self.database = settings.mysql.database
        
        self._initialize_pool()
        self._initialized = True

    def _bootstrap_database(self) -> bool:
        """Create the target database if it does not exist in XAMPP MySQL."""
        conn = None
        cursor = None
        try:
            logger.info(f"Connecting to MySQL server at {self.host}:{self.port} to bootstrap database '{self.database}'...")
            conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            conn.commit()
            logger.info(f"Database '{self.database}' verified/created successfully.")
            return True
        except mysql.connector.Error as err:
            logger.error(f"Failed to bootstrap MySQL database '{self.database}': {err}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _initialize_pool(self):
        """Initialize the connection pool, bootstrapping database if needed."""
        db_config = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }
        
        try:
            logger.info(f"Initializing MySQL Connection Pool (Host: {self.host}, Database: {self.database})")
            self._pool = pooling.MySQLConnectionPool(
                pool_name="placement_pool",
                pool_size=5,
                **db_config
            )
            logger.info("MySQL Connection Pool initialized successfully.")
        except mysql.connector.Error as err:
            # Error 1049 is "Unknown database"
            if err.errno == 1049:
                logger.warning(f"Database '{self.database}' not found. Attempting auto-bootstrap...")
                if self._bootstrap_database():
                    # Retry pool initialization after database creation
                    try:
                        self._pool = pooling.MySQLConnectionPool(
                            pool_name="placement_pool",
                            pool_size=5,
                            **db_config
                        )
                        logger.info("MySQL Connection Pool initialized successfully after auto-bootstrap.")
                        return
                    except mysql.connector.Error as retry_err:
                        logger.error(f"Failed to initialize pool after bootstrapping: {retry_err}")
            else:
                logger.error(f"Failed to initialize MySQL Connection Pool: {err}")
            self._pool = None

    def get_connection(self):
        """Retrieve a thread-safe connection from the pool.
        
        Returns:
            PooledMySQLConnection: A pooled MySQL connection instance.
            
        Raises:
            mysql.connector.Error: If the pool is not initialized or fails to yield a connection.
        """
        if not self._pool:
            # Lazy retry pool initialization
            self._initialize_pool()
            if not self._pool:
                raise mysql.connector.Error(msg="MySQL Connection Pool is offline or uninitialized.")
        
        try:
            return self._pool.get_connection()
        except mysql.connector.Error as err:
            logger.error(f"Error fetching connection from pool: {err}")
            raise

    def execute_query(self, query: str, params: tuple = None, is_select: bool = True):
        """Execute a query securely using a pooled connection.
        
        Args:
            query: SQL query to run.
            params: Tuple of query parameters (for SQL injection prevention).
            is_select: True if retrieving rows, False if writing/updating.
            
        Returns:
            List[Dict] for SELECT queries; Row count (int) for INSERT/UPDATE/DELETE.
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            # Returns dictionary results instead of tuples for clean service-based usage
            cursor = conn.cursor(dictionary=True)
            
            if params is not None:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if is_select:
                result = cursor.fetchall()
                return result
            else:
                conn.commit()
                return cursor.rowcount
        except mysql.connector.Error as err:
            logger.error(f"Database query execution failed: {err}\nQuery: {query}\nParams: {params}")
            if conn and not is_select:
                try:
                    conn.rollback()
                except Exception as rollback_err:
                    logger.error(f"Rollback failed: {rollback_err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close() # Return connection to the pool
