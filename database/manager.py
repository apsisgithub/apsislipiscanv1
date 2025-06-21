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


def check_existing_user(db: 'SQLDatabaseManager', platform: str, platform_identifier: str) -> pd.DataFrame:
    """
    Checks if a user exists based on their platform and platform-specific unique identifier.

    - For 'google', platform_identifier is the user's email address (checked against 'user_email' column).
    - For 'apple', platform_identifier is the user's Apple User ID (checked against 'userIdentifier' column).

    Args:
        db (SQLDatabaseManager): The database manager instance.
        platform (str): The platform of the user (e.g., 'google', 'apple').
        platform_identifier (str): The unique identifier for the user on that platform.

    Returns:
        pd.DataFrame: DataFrame containing the user row if found (should be 0 or 1 row
                      due to unique constraints), else an empty DataFrame.
    """
    logger.info(f"Checking for existing user on platform: '{platform}' with platform-specific identifier: '{platform_identifier}'")

    condition_dict = {"platform": platform}
    identifier_column_for_query = ""

    # Normalize platform for comparison
    normalized_platform = platform.lower()

    if normalized_platform == "google":
        identifier_column_for_query = "user_email"
        condition_dict[identifier_column_for_query] = platform_identifier
    elif normalized_platform == "apple":
        identifier_column_for_query = "userIdentifier"
        condition_dict[identifier_column_for_query] = platform_identifier
    else:
        logger.warning(f"Unsupported platform '{platform}' for user check. Returning empty DataFrame.")
        return pd.DataFrame()

    df = db.user_table.select(condition_dict=condition_dict)

    if df.empty:
        logger.info(f"No user found on platform '{platform}' with {identifier_column_for_query}: '{platform_identifier}'")
    else:
        if len(df) > 1:
            logger.warning(
                f"Found {len(df)} users for platform '{platform}' with {identifier_column_for_query}: '{platform_identifier}'. "
                f"This indicates a potential issue with data integrity or unique constraint enforcement. "
                f"Using the first record. User IDs: {df['user_id'].tolist() if 'user_id' in df.columns else 'N/A'}"
            )
        elif 'user_id' in df.columns:
             logger.info(
                f"Found user on platform '{platform}' with {identifier_column_for_query}: '{platform_identifier}'. "
                f"User ID: {df.iloc[0]['user_id']}"
            )
        else: # Should not happen if select returns data from UserTable
            logger.warning(
                f"Found user on platform '{platform}' with {identifier_column_for_query}: '{platform_identifier}', but 'user_id' column is missing in the result."
            )
    return df


def new_user_insert(
    db: 'SQLDatabaseManager',
    platform: str,
    user_name: str,       # Must be provided for new user
    user_email: str,      # Must be provided for new user
    db_user_identifier: str # The value to store in UserTable.userIdentifier
) -> int:
    """
    Inserts a new user into the user_table.
    Expects all necessary non-nullable fields to be provided.

    Args:
        db (SQLDatabaseManager): The database manager instance.
        platform (str): The platform ('google', 'apple').
        user_name (str): The name of the new user.
        user_email (str): The email address of the new user.
        db_user_identifier (str): The value for the 'userIdentifier' DB column.
                                   (For Google, this will be their email; for Apple, their Apple ID).
    Returns:
        int: Returns 0 if successful. Exits on error via db.user_table.insert.
    """
    try:
        user_id = str(uuid.uuid4())
        registered_time = datetime.now() 

        if not all([platform, user_name, user_email, db_user_identifier]):
             logger.error("new_user_insert: Missing one or more required fields for new user creation "
                          f"(platform, user_name, user_email, db_user_identifier). platform='{platform}', "
                          f"name='{user_name}', email='{user_email}', db_id='{db_user_identifier}'")
             
        insert_data_list = [
            {
                "user_id": user_id,
                "platform": platform.lower(), # Normalize
                "userIdentifier": db_user_identifier,
                "user_name": user_name,
                "user_email": user_email,
                "registered": registered_time,
                "last_login": None, # For a new user created via this specific function.
                                    # handle_user_strict_previous_login will set DB last_login to current time.
                "remark": None
            }
        ]

        logger.info(f"Preparing to insert new user via new_user_insert: user_id='{user_id}', platform='{platform}', "
                    f"DB_userIdentifier='{db_user_identifier}', user_email='{user_email}', name='{user_name}'")

        
        return_code = db.user_table.insert(insert_data_list)

        if return_code == 0:
            logger.info(f"Successfully inserted new user with user_id='{user_id}' via new_user_insert.")
        return return_code

    except Exception as e:
        logger.error(f"An unexpected error occurred in new_user_insert for platform '{platform}', email '{user_email}': {e}")
        sys.exit(-1)  # Exit on error, as per original spec


def handle_user_strict_previous_login(
    db: 'SQLDatabaseManager',
    client_platform: str,
    client_user_email: str, # Email from client, can be None for Apple reinstall
    client_user_name: str ,  # Name from client, can be None for Apple reinstall
    client_userIdentifier: str  # Apple User ID from client, None for Google
) -> dict:
    """
    Handles user login/registration with strict "last_login means previous login" rule.

    Args:
        db: SQLDatabaseManager instance.
        client_platform (str): 'google' or 'apple'.
        client_user_email (str ): Email from client.
        client_user_name (str ): Name from client.
        client_userIdentifier (str ): Apple User ID from client.

    Returns:
        dict: User information and 'user_type'.
    """
    current_session_login_time = datetime.now() 
    normalized_platform = client_platform.lower()

    logger.info(f"Handling user login: platform='{normalized_platform}', client_email='{client_user_email}', "
                f"client_name='{client_user_name}', client_apple_id='{client_userIdentifier}' at {current_session_login_time}")

    identifier_for_check = ""
    db_user_identifier_value_for_new_user = "" # Value for UserTable.userIdentifier column if new user

    if normalized_platform == "google":
        if not client_user_email: # Google MUST provide email
            logger.error("Google login attempt without email.")
            return {"error": "Email is required for Google login.", "user_type": "error"}
        identifier_for_check = client_user_email
        db_user_identifier_value_for_new_user = client_user_email # For Google, email is also the DB userIdentifier
    elif normalized_platform == "apple":
        if not client_userIdentifier: # Apple MUST provide userIdentifier
            logger.error("Apple login attempt without userIdentifier.")
            return {"error": "userIdentifier is required for Apple login.", "user_type": "error"}
        identifier_for_check = client_userIdentifier
        db_user_identifier_value_for_new_user = client_userIdentifier # For Apple, this is the Apple ID
    else:
        logger.error(f"Unsupported platform: {normalized_platform}")
        return {"error": f"Unsupported platform: {normalized_platform}", "user_type": "error"}

    existing_user_df = check_existing_user(db, normalized_platform, identifier_for_check)

    if not existing_user_df.empty:
        # ----- USER EXISTS -----
        if len(existing_user_df) > 1:
            logger.warning(f"Multiple users found for platform='{normalized_platform}', id_check='{identifier_for_check}'. Using first record.")
        
        user_data_row = existing_user_df.iloc[0]
        user_id_from_db = user_data_row['user_id']
        logger.info(f"User found: user_id='{user_id_from_db}'. Processing as existing user.")

        actual_previous_login_time_for_return = user_data_row['last_login']
        user_info_to_return = user_data_row.to_dict() # Start with all DB data

        # Update last_login in DB to current time
        pk_for_update = {"user_id": user_id_from_db}
        db.user_table.update_one_cell(
            column_name="last_login",
            new_value=current_session_login_time,
            primary_key_values=pk_for_update
        )
        logger.info(f"Updated DB last_login for user_id='{user_id_from_db}' to {current_session_login_time}.")

        # For existing users, client_user_name and client_user_email might be new/updated
        # or absent (e.g. Apple reinstall). Only update if provided by client.
        updates_for_existing_user = {}
        
        if client_user_name is not None and user_info_to_return.get("user_name") != client_user_name:
            updates_for_existing_user["user_name"] = client_user_name
            user_info_to_return["user_name"] = client_user_name # Reflect update in return
            logger.info(f"Updating user_name for user_id='{user_id_from_db}' to '{client_user_name}'.")
        
        if client_user_email is not None and user_info_to_return.get("user_email") != client_user_email:
            # Special care for Google: user_email is part of unique key and identifier_for_check.
            # Changing it for an existing Google user based on an old email lookup could be problematic.
            # For Apple, email can be updated more freely if provided.
            if normalized_platform == "apple":
                updates_for_existing_user["user_email"] = client_user_email
                user_info_to_return["user_email"] = client_user_email # Reflect update in return
                logger.info(f"Updating user_email for Apple user_id='{user_id_from_db}' to '{client_user_email}'.")
            elif normalized_platform == "google" and identifier_for_check == client_user_email:
                # If it's Google and the new email is the same as the one we looked up, no change needed.
                # If they are trying to *change* their email, that's a more complex flow not handled here.
                pass
            else: # Google user, client_user_email is different from lookup email
                 logger.warning(f"Google user '{identifier_for_check}' attempted to provide a different email "
                                f"'{client_user_email}'. Email update for Google primary identifier not supported in this flow.")


        if updates_for_existing_user:
            db.user_table.update(condition_columns=["user_id"], update_array=[{"user_id": user_id_from_db, **updates_for_existing_user}])

        user_info_to_return["user_type"] = "existing"
        user_info_to_return["last_login"] = actual_previous_login_time_for_return # This is the key change for "existing"

        logger.info(f"Returning existing user data for user_id='{user_id_from_db}'. Actual previous login: {actual_previous_login_time_for_return}")
        return user_info_to_return

    else:
        # ----- NEW USER -----
        logger.info(f"User not found with platform='{normalized_platform}', id_check='{identifier_for_check}'. Creating new user.")
        
        # For a new user, we NEED user_name and user_email (as they are NOT NULL in DB)
        if client_user_name is None:
            logger.error(f"Cannot create new {normalized_platform} user: user_name not provided by client for id_check='{identifier_for_check}'.")
            return {"error": "user_name is required to create a new user account.", "user_type": "error"}
        if client_user_email is None:
            logger.error(f"Cannot create new {normalized_platform} user: user_email not provided by client for id_check='{identifier_for_check}'.")
            return {"error": "user_email is required to create a new user account.", "user_type": "error"}

        new_user_id = str(uuid.uuid4())
        
        new_user_data_for_db_insert = {
            "user_id": new_user_id,
            "platform": normalized_platform,
            "userIdentifier": db_user_identifier_value_for_new_user,
            "user_name": client_user_name,       # Use name from client
            "user_email": client_user_email,     # Use email from client
            "registered": current_session_login_time,
            "last_login": current_session_login_time, # DB last_login is current time for new user
            "remark": None
        }
        
        # Directly call db.user_table.insert here
        return_code = db.user_table.insert([new_user_data_for_db_insert])
        
        if return_code != 0: # Should not happen if SQLTable.insert calls sys.exit
            logger.error(f"Failed to insert new user (platform='{normalized_platform}', id_check='{identifier_for_check}'). Insert returned: {return_code}")
            return {"error": "Failed to create new user record in database.", "user_type": "error"}

        logger.info(f"New user created: user_id='{new_user_id}', platform='{normalized_platform}', "
                    f"DB_userIdentifier='{db_user_identifier_value_for_new_user}', email='{client_user_email}'. "
                    f"DB last_login set to {current_session_login_time}.")
        
        output_data_for_new_user = new_user_data_for_db_insert.copy()
        output_data_for_new_user["last_login"] = None # Output last_login is None for new user, as per original spec
        output_data_for_new_user["user_type"] = "new"
        
        logger.info(f"Returning new user data for user_id='{new_user_id}'. Output last_login is None.")
        return output_data_for_new_user