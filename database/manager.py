import numpy as np
import urllib.parse
import pandas as pd
from loguru import logger
import uuid
from datetime import datetime
import sys
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session
from psycopg2.extensions import register_adapter, AsIs
from .definitions import (Base,SQLTable,UserTable)
""" psycopg2 throws datatype error into postgres DB.
Following block of code can solve this issue.
Source: https://stackoverflow.com/a/56390591
"""
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.int64, addapt_numpy_int64)

class SQLDatabaseManager:
    """SQL database manager. It will save tracklet informations in SQL DB
    """
    def __init__(self, database_config: dict,create_db :bool=False) -> None:
        """create all the SQL tables with appropriate column names.
        Args:
                database_config (dict): config dictionary containing information about databases
                
        """
        self.create_db = create_db
        self.config = database_config
        self.create_engine(database_config)
        self.declare_tables()
        
    def database_exists(self, user: str, password: str, host: str, port: int, database_name: str) -> bool:
        """
        Check if a PostgreSQL database exists.
        """
        try:
            conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
            conn.autocommit = True
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (database_name,))
            exists = cursor.fetchone() is not None

            cursor.close()
            conn.close()
            return exists
        except psycopg2.Error as e:
            logger.error("Error checking database existence: {}".format(e))
            sys.exit(-1)

    def drop_database(self, user: str, password: str, host: str, port: int, database_name: str):
        """
        Drop (delete) the PostgreSQL database if it exists.
        """
        try:
            # Connect to the 'postgres' database (default maintenance database)
            conn = psycopg2.connect(
                dbname="postgres",
                user=user,
                password=password,
                host=host,
                port=port
            )
            conn.autocommit = True  # Necessary for DROP DATABASE and connection termination
            cursor = conn.cursor()

            # Terminate all active connections to the target database
            terminate_query = sql.SQL("""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = %s
                AND pid <> pg_backend_pid();
            """)
            cursor.execute(terminate_query, (database_name,))
            logger.info("Terminated all active connections to the database: {}".format(database_name))

            # Drop the database
            drop_query = sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name))
            cursor.execute(drop_query)
            logger.info("Database {} dropped successfully.".format(database_name))

            cursor.close()
            conn.close()

        except psycopg2.Error as e:
            logger.error("Error dropping database {}: {}".format(database_name, e))
            sys.exit(-1)
        except Exception as e:
            logger.error("An unexpected error occurred while dropping the database: {}".format(e))
            sys.exit(-1)


    def create_database(self, user: str, password: str, host: str, port: int, database_name: str):
        """
        Create a new PostgreSQL database.
        """
        try:
            conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
            conn.autocommit = True
            cursor = conn.cursor()

            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            logger.info("Database {} created successfully.".format(database_name))

            cursor.close()
            conn.close()
        except psycopg2.Error as e:
            logger.error("Error creating database {}: {}".format(database_name, e))  
            sys.exit(-1) 

    def create_engine(self, database_config) -> None:
        """create SQL engine.

        Args:
            database_config : user, password, localhost, port and DB name of the postgres database
        Returns:
            None
        """
        try:
            conn_url = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
                database_config['user'],
                database_config['password'],
                database_config['host'],
                database_config['port'],
                database_config['database']
            )

            self.engine = create_engine(conn_url, echo=False)

        except Exception as exc:
            logger.error("Exception occurred while creating SQL engine. Error: {}".format(exc))
            sys.exit(-1)


    def declare_tables(self) -> None:
        """Declare the SQL tables.

        Args:
            self
        Returns:
            None
        """
        try:
            if self.create_db:
                Base.metadata.create_all(self.engine)
            self.user_table = SQLTable(self.engine, UserTable.__table__)
        except Exception as exc:
            logger.error('Exception occured while table defining. Error: {}'.format(exc))
            sys.exit(-1)
        
    def user_table_insert(self, insert_data: list[dict]) -> int:
        """
        Insert data into the user_table.

        Args:
            insert_data (list[dict]): A list of dictionaries, where each dictionary
                                      represents a row to be inserted. Column names
                                      are keys and their values are the data.
        Returns:
            int: Returns 0 if successful, otherwise an error might lead to sys.exit
                 via the underlying SQLTable.insert method.
        """
        logger.info(f"Attempting to insert {len(insert_data)} rows into {self.user_table.table.name}.")
        if not hasattr(self, 'user_table'):
            logger.error("User table is not initialized in SQLDatabaseManager.")
            sys.exit(-1)

        return_code = self.user_table.insert(insert_data)

        if return_code == 0:
            logger.info(f"Successfully inserted data into {self.user_table.table.name}.")
        else:
            # This path might not be reached if SQLTable.insert exits on error
            logger.error(f"Failed to insert data into {self.user_table.table.name}.")
        return return_code
    

def sql_table_names(engine):
    """get SQL table names from the database
    Args:
        engine: sql engine
    Returns:
        str: returns table names
    """
    insp = inspect(engine)
    table_names = insp.get_table_names()
    logger.info("SQL table names: {}".format(table_names))
    return table_names


def check_existing_user(db: SQLDatabaseManager, user_email: str) -> pd.DataFrame:
    """
    Checks if a user with the specified email address exists in the user_table.

    This function queries the 'user_table' in the database managed by the 'db'
    instance. It specifically looks for rows where the 'user_email' column
    matches the provided 'user_email' string.

    Args:
        db (SQLDatabaseManager): An instance of the SQLDatabaseManager class,
            which provides access to the database and its tables. It is expected
            that `db.user_table` is an initialized SQLTable instance representing
            the 'user_table'.
        user_email (str): The email address to search for in the 'user_email'
            column of the 'user_table'.

    Returns:
        pd.DataFrame:
            - If one or more users are found with the given 'user_email', the
              DataFrame will contain the full row(s) for those users from the
              'user_table'.
            - If no user is found with the given 'user_email', an empty
              Pandas DataFrame (with columns defined but no rows) is returned.
            - In case of a database error during the select operation, the
              underlying `SQLTable.select` method is designed to log the error
              and call `sys.exit(-1)`, so the program would terminate before
              this function returns in such a scenario.
    """
    # 1. Accessing the User Table Object:
    #   `db` is an instance of `SQLDatabaseManager`.
    #   `db.user_table` is an attribute of `SQLDatabaseManager` that holds an
    #   instance of `SQLTable`. This `SQLTable` instance is specifically configured
    #   to operate on the database table defined by `UserTable.__table__` (which is "user_table").
    #   So, `db.user_table` gives us the object to interact with the "user_table".

    # 2. Calling the `select` Method:
    #   `db.user_table.select(...)` calls the `select` method of the `SQLTable` class.
    #   This method is designed to retrieve data from the associated SQL table.

    # 3. Specifying the Condition:
    #   `condition_dict={"user_email": user_email}` is passed as an argument.
    #   The `SQLTable.select` method interprets this dictionary to build a
    #   SQL WHERE clause. For this specific input, it will construct a condition
    #   equivalent to: `WHERE user_table.user_email = 'the_value_of_user_email_variable'`
    #
    #   Internally, `SQLTable.select` does something like this (simplified):
    #   ```python
    #   # Inside SQLTable.select method:
    #   # stmt_tc = None
    #   # for col, val in condition_dict.items(): # col is "user_email", val is the actual email string
    #   #     stmt_c = getattr(self.table.c, col) == val  # self.table.c.user_email == 'the_email@example.com'
    #   #     if stmt_tc is None:
    #   #         stmt_tc = select(self.table).where(stmt_c)
    #   #     else:
    #   #         stmt_tc = stmt_tc.where(stmt_c)
    #   # df = pd.read_sql(stmt_tc, conn)
    #   ```

    logger.info(f"Checking for existing user with email: {user_email}")
    df = db.user_table.select(condition_dict={"user_email": user_email})

    # 4. Result Processing by `SQLTable.select`:
    #   The `SQLTable.select` method executes the constructed SQL query against
    #   the database using `pd.read_sql(stmt_tc, conn)`.
    #   - If the query finds matching rows (i.e., users with that email),
    #     `pd.read_sql` returns a DataFrame populated with those rows.
    #   - If no rows match the condition, `pd.read_sql` returns an empty
    #     DataFrame (it will have the correct column names from `user_table`
    #     but 0 rows).

    # 5. Returning the DataFrame:
    #   The DataFrame `df` obtained from the `select` call is then returned by
    #   `check_existing_user`.
    if df.empty:
        logger.info(f"No user found with email: {user_email}")
    else:
        logger.info(f"Found {len(df)} user(s) with email: {user_email}. User IDs: {df['user_id'].tolist() if 'user_id' in df else 'N/A'}")
        # Note: UserTable has a composite primary key (user_id, user_email).
        # This means multiple distinct 'user_id' values could theoretically be associated
        # with the same 'user_email' if that's how data was inserted, and all such
        # records would be returned. A common application design might enforce
        # user_email uniqueness separately if a single user account per email is desired.

    return df


def new_user_insert(db: SQLDatabaseManager, user_email: str, user_name: str) -> int:
    """
    Inserts a new user into the user_table with a unique ID and registration timestamp.

    Args:
        db (SQLDatabaseManager): The database manager instance.
        user_email (str): The email of the new user. This is part of the composite primary key.
        user_name (str): The name of the new user.
    
    Returns:
        int: Returns 0 if successful. The underlying insert method (db.user_table_insert)
             will cause sys.exit(-1) on database errors (e.g., integrity violations for PK).
    """
    try:
        # Generate a unique user_id using UUID
        user_id = str(uuid.uuid4())
        
        # Get the current time for the 'registered' field
        registered_time = datetime.now()

        # Prepare the data for insertion.
        # 'last_login' and 'remark' are nullable and can be set to None for a new user.
        insert_data = [
            {
                "user_id": user_id,
                "user_name": user_name,
                "user_email": user_email,
                "registered": registered_time,
                "last_login": None,  # New user has not logged in yet
                "remark": None       # No initial remark
            }
        ]
        
        logger.info(f"Preparing to insert new user: email='{user_email}', name='{user_name}', generated user_id='{user_id}'")
        
        # Call the user_table_insert method of SQLDatabaseManager.
        # This method handles its own logging for success/failure and will call sys.exit
        # on critical database errors (like primary key violations if the (user_id, user_email) pair already exists,
        # or other database connection/operation issues).
        return_code = db.user_table_insert(insert_data)
        
        return return_code

    except Exception as e:
        # This catch block is primarily for unexpected errors occurring *before* the call
        # to db.user_table_insert (e.g., if uuid.uuid4() or datetime.now() failed, though highly unlikely).
        # Database operation errors are handled within db.user_table_insert -> SQLTable.insert.
        logger.error(f"An unexpected error occurred in new_user_insert logic for email '{user_email}' before database operation: {e}")
        # Consistent with other error handling in SQLTable, exiting here might be expected.
        sys.exit(-1)


def handle_user_strict_previous_login(db: SQLDatabaseManager, user_email: str, user_name: str) -> dict:
    """
    Handles a user session based on a strict "last_login means previous login" rule.

    - If the user (identified by user_email) exists:
        - Fetches the current 'last_login' from DB (this is the actual previous login).
        - Updates their 'last_login' timestamp in the DB to the current time (this current
          login time will serve as the 'last_login' for their *next* session).
        - Returns a dictionary with all database fields for the (first found) user.
          The 'last_login' in this returned dict is the one fetched *before* the update
          (i.e., the timestamp of their actual previous login). 'user_type' is 'existing'.
    - If the user does not exist:
        - Creates a new user record with a unique 'user_id'.
        - 'registered' is set to current_time.
        - 'last_login' in the DB is set to the current_time (this will be the 'previous login'
           if they log in again; or None, see note below).
        - Returns a dictionary with all fields of the new user, 'user_type': 'new'.
          'last_login' in the returned dict is None.

    Args:
        db (SQLDatabaseManager): The database manager instance.
        user_email (str): The email of the user.
        user_name (str): The name of the user.

    Returns:
        dict: A dictionary containing user information and 'user_type'.
    """
    current_session_login_time = datetime.now() # Use timezone-aware datetime
    logger.info(f"Handling user request (strict prev login) for email: {user_email}, name: {user_name} at {current_session_login_time}")

    existing_user_df = check_existing_user(db, user_email)

    if not existing_user_df.empty:
        # User Exists
        logger.info(f"User with email '{user_email}' found. Processing as existing user (strict prev login).")

        # We'll process the first record found for the return value,
        # but update all records matching the email.
        
        actual_previous_login_time_for_return = None # Initialize
        first_record_processed = False
        user_info_to_return = {}

        for index, row_data in existing_user_df.iterrows():
            user_id_from_db = row_data['user_id']
            
            # Fetch the actual previous login time from this specific record
            # The 'last_login' column in UserTable should be DateTime, nullable=True
            db_last_login_value = row_data['last_login']

            if not first_record_processed:
                actual_previous_login_time_for_return = db_last_login_value
                # Prepare the base of the user_info_to_return from the first record
                user_info_to_return = row_data.to_dict()
                first_record_processed = True
            
            # Update the 'last_login' in the DB for this record to the current session's login time.
            # This current_session_login_time will become the 'previous_login_time' for the *next* login.
            primary_key_values = {"user_id": user_id_from_db, "user_email": user_email}
            db.user_table.update_one_cell(
                column_name="last_login",
                new_value=current_session_login_time,
                primary_key_values=primary_key_values
            )
            logger.info(f"Updated DB last_login for user_id: {user_id_from_db}, email: {user_email} to {current_session_login_time} (this session's time). Previous was: {db_last_login_value}")

        # Finalize the dictionary to return using data from the first matched record
        user_info_to_return["user_type"] = "existing"
        user_info_to_return["last_login"] = actual_previous_login_time_for_return # This is the key change for "existing"

        if user_info_to_return.get("user_name") != user_name:
            logger.warning(
                f"Provided user_name '{user_name}' differs from DB user_name "
                f"'{user_info_to_return.get('user_name')}' for user_id: {user_info_to_return.get('user_id')}, email: {user_email}. "
                f"DB name not updated."
            )

        logger.info(f"Returning existing user data for email: {user_email}, user_id: {user_info_to_return.get('user_id')}. Actual previous login: {actual_previous_login_time_for_return}")
        return user_info_to_return

    else:
        # User Does Not Exist - Create a new user
        logger.info(f"User with email '{user_email}' not found. Creating new user (strict prev login).")
        
        new_user_id = str(uuid.uuid4())
        
        # For a brand new user:
        # - 'registered' is current_session_login_time.
        # - 'last_login' in the DB:
        #   Option 1: Set to current_session_login_time. If they log in again, this becomes their "previous login".
        #   Option 2: Set to None. If they log in again, their "previous login" would be None.
        #   The prompt "set current login as last login" (for the DB) from the *previous* interpretation
        #   points towards Option 1 for consistency. Let's stick with that for DB.
        
        new_user_data_for_db = {
            "user_id": new_user_id,
            "user_name": user_name,
            "user_email": user_email,
            "registered": current_session_login_time,
            "last_login": current_session_login_time, # This session's time, will be "previous" on next login
            "remark": None
        }
        
        return_code = db.user_table.insert([new_user_data_for_db])
        
        if return_code != 0:
            logger.error(f"Failed to insert new user (strict) with email: {user_email}. Insert returned: {return_code}")
            return {"error": "Failed to create new user", "user_type": "error"}

        logger.info(f"New user created with user_id: {new_user_id}, email: {user_email}. DB last_login set to {current_session_login_time}")
        
        # Prepare the dictionary to return for the new user
        output_data_for_new_user = new_user_data_for_db.copy()
        output_data_for_new_user["last_login"] = None # For a new user, the "previous" login is None in the output.
        output_data_for_new_user["user_type"] = "new"
        
        logger.info(f"Returning new user data for email: {user_email}, user_id: {new_user_id} (output last_login is None)")
        return output_data_for_new_user