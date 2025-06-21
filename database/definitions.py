import sys
import pandas as pd
from sqlalchemy import (select, insert, delete, update, 
                        Column, Integer, String, Float, 
                        LargeBinary, DateTime, Boolean, 
                        PrimaryKeyConstraint, Index,  
                        ForeignKeyConstraint,
                        UniqueConstraint,text)

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import and_, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.postgresql import ARRAY  # Added for list type support in PostgreSQL

from loguru import logger


class Base(DeclarativeBase):
    """ Declarative Base refers to a MetaData collection
    new classes that are subclasses of Base, combined with appropriate class-level directives, 
    they will each be established as a new ORM mapped class at class creation time.
    """
    pass

class UserTable(Base):
    __tablename__ = "user_table"
    
    user_id = Column(String, nullable=False)
    platform= Column(String, nullable=False)
    userIdentifier   = Column(String, nullable=False)
    user_name = Column(String,nullable=False)
    user_email= Column(String,nullable=False)
    registered= Column(DateTime,nullable=False)
    last_login= Column(DateTime,nullable=True)
    remark    = Column(String, nullable=True)  
    
    __table_args__ = (
        PrimaryKeyConstraint('user_id', name='pk_user_table'),
        # Conditional uniqueness using Index
        Index('uq_google_user_email', # Index name
              'user_email',          # Column(s) in the index
              unique=True,
              postgresql_where=text("platform = 'google'")),
        Index('uq_apple_user_identifier', # Index name
              'userIdentifier',           # Column(s) in the index
              unique=True,
              postgresql_where=text("platform = 'apple'")),
    )

    def __repr__(self) -> str:
        return (
            f"<UserTable(user_id={self.user_id!r}, platform={self.platform!r}, "
            f"user_name={self.user_name!r}, user_email={self.user_email!r}, "
            f"userIdentifier={self.userIdentifier!r})>"
        )


class SQLTable:
    """SQL table class to handle manipulating data to SQL Database. 
    """
    def __init__(self, engine, table) -> None:
        """Initialize parameters for table operation
        """
        
        self.engine = engine
        self.table = table
        

    def create(self, table_name: str, header_columns: list) -> int:
        """create a blank SQL table

        Args:
            table_name (str): sql table name
            header_columns (list): header columns for the sql table
        
        Returns:
            int: returns 0 if successful
        """
        # TODO
        pass


    def insert(self, insert_data: list[dict]) -> int:
        """insert new data to the SQL table with insert_data

        Args:
            insert_data (list of dict): list of dictionary of column and data pair to be inserted

        
        Returns:
            int: returns 0 if successful
        """

        # if no values are present
        if not len(insert_data):
            return
        
        try:
            with self.engine.connect() as conn:
                _ = conn.execute(
                    insert(self.table),
                    insert_data,
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"An error occurred during INSERT: {exc}")
            sys.exit(-1)
        finally:
            conn.close()
        
        return 0
    
    
        
    def select(self, condition_dict: dict = None, range_condition_dict: dict = None) -> pd.DataFrame:
        """select rows of data from the database

        Args:
            condition_dict (dict, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: dataframe of the SQL table satisfying input condition_dict.
        """
        try:
            if condition_dict:
                with self.engine.connect() as conn:
                    stmt_tc = None
                    for col, val in condition_dict.items():
                        stmt_c = getattr(self.table.c, col) == val
                        if stmt_tc == None:
                            stmt_tc = select(self.table).where(stmt_c)
                        else:
                            stmt_tc = stmt_tc.where(stmt_c)
                    # Read SQL query or database table into a DataFrame.
                    df = pd.read_sql(stmt_tc, conn)
            # Apply range conditions if provided
            elif range_condition_dict:
                with self.engine.connect() as conn:
                    stmt_tc = None
                    for col, (lower, upper) in range_condition_dict.items():
                        stmt_c = getattr(self.table.c, col).between(lower, upper)
                        if stmt_tc == None:
                            stmt_tc = select(self.table).where(stmt_c)
                        else:
                            stmt_tc = stmt_tc.where(stmt_c)
                    # Read SQL query or database table into a DataFrame.
                    df = pd.read_sql(stmt_tc, conn)
            else:
                # return all the data
                with self.engine.connect() as conn:
                    stmt_tc = select(self.table)
                    # Read SQL query or database table into a DataFrame.
                    df = pd.read_sql(stmt_tc, conn)
        except Exception as exc:
            logger.error(f"An error occurred during SELECT: {exc}")
            sys.exit(-1)
        finally:
            conn.close()
            
        return df

    
    def multi_select(self, condition_list: list = None, range_condition_list: list = None) -> pd.DataFrame:
        """Select rows from the database based on multiple condition sets.

        Args:
            condition_list (list, optional): List of dictionaries with conditions to filter data using AND within each dictionary and OR across dictionaries.
            range_condition_list (list, optional): List of dictionaries with range conditions to apply similarly.

        Returns:
            pd.DataFrame: Dataframe containing the filtered results.
        """
        try:
            with self.engine.connect() as conn:
                stmt_tc = select(self.table)
                conditions = []
                
                # Process equality conditions
                if condition_list:
                    for condition_dict in condition_list:
                        sub_conditions = [getattr(self.table.c, col) == val for col, val in condition_dict.items()]
                        conditions.append(and_(*sub_conditions))
                
                # Process range conditions
                if range_condition_list:
                    for range_dict in range_condition_list:
                        sub_conditions = [getattr(self.table.c, col).between(lower, upper) for col, (lower, upper) in range_dict.items()]
                        conditions.append(and_(*sub_conditions))
                
                # Apply OR across different condition sets
                if conditions:
                    stmt_tc = stmt_tc.where(or_(*conditions))
                    # Execute query and return results as DataFrame
                    df = pd.read_sql(stmt_tc, conn)
                else:
                    df=pd.DataFrame({})
        except Exception as exc:
            logger.error(f"An error occurred during SELECT: {exc}")
            sys.exit(-1)
        finally:
            conn.close()
            
        return df


    def update(self, condition_columns: list, update_array: list[dict] = None) -> int:
        """update a row of a SQL table.
        Args:
            condition_columns (list): list of column names that will be used to primary key for updating rows
            update_array (list of dict): list of dictionary of column and data pair to be inserted

        Returns:
            int: returns 0 if successful
        """
        
        if not len(update_array):
            return
        try:
            with self.engine.connect() as conn:
                for value in update_array:
                    stmt_tc = None
                    for col in condition_columns:
                        stmt_c = getattr(self.table.c, col) == value[col]
                        if stmt_tc == None:
                            stmt_tc = update(self.table).where(stmt_c)
                        else:
                            stmt_tc = stmt_tc.where(stmt_c)
                    _ = conn.execute(
                        stmt_tc.values(value)
                    )
                    conn.commit()
        except Exception as exc:
            logger.error(f"An error occurred during UPDATE: {exc}")
            sys.exit(-1)
        finally:
            conn.close()
            
        return 0

    def upsert(self, insert_data: list[dict], update_columns: list[str]) -> int:
        """
        Insert new rows or update an existing row's single column if conflict occurs.

        Args:
            insert_data (list[dict]): List of data dictionaries to insert or update.
            update_column (list[str]): The list of columns name to update on conflict.

        Returns:
            int: Returns 0 if successful.
        """
        if not insert_data:
            return 0

        try:
            with self.engine.connect() as conn:
                # Infer conflict columns (primary keys)
                primary_keys = [col.name for col in self.table.primary_key]

                stmt = pg_insert(self.table).values(insert_data)
                # Dynamically construct the SET clause for ON CONFLICT DO UPDATE
                update_dict = {col: getattr(stmt.excluded, col) for col in update_columns}


                stmt = stmt.on_conflict_do_update(
                    index_elements=primary_keys,
                    set_=update_dict
                )
                conn.execute(stmt)
                conn.commit()

        except Exception as exc:
            logger.error(f"An error occurred during UPSERT: {exc}")
            sys.exit(-1)
        finally:
            conn.close()
        
        return 0

        
    def update_only_one_column(
        self, 
        column_name: str, 
        new_values: list, 
        primary_key_values: list[dict]
    ) -> int:
        """
        Update a single column in the SQL table for given primary key values.

        Args:
            column_name (str): The name of the column to update.
            new_values (list): The new values to set for the column.
            primary_key_values (List[dict]): A list of dictionaries mapping primary key column names to their respective values.

        
        Returns:
            int: Returns 0 if the update is successful.
        """
        try:
            
            # Prepare the update statements
            with self.engine.connect() as conn:
                for pk_values, new_value in zip(primary_key_values, new_values):
                    stmt = update(self.table).where(
                        *[
                            getattr(self.table.c, pk_column) == pk_value
                            for pk_column, pk_value in pk_values.items()
                        ]
                    ).values({column_name: new_value})
                    conn.execute(stmt)
                conn.commit()

            return 0
        except Exception as exc:
            logger.error(f"An error occurred: {exc}")
            sys.exit(-1)
        finally:
            conn.close()
    
    def update_one_cell(
        self, 
        column_name: str, 
        new_value, 
        primary_key_values: dict
    ) -> int:
        """
        Update a single cell in the SQL table for a given primary key.

        Args:
            column_name (str): The name of the column to update.
            new_value: The new value to set for the column.
            primary_key_values (dict): A dictionary mapping primary key column names to their respective values.

    
        Returns:
            int: Returns 0 if the update is successful.
        """
        try:
            # Prepare the update statement
            with self.engine.connect() as conn:
                stmt = (
                    update(self.table)
                    .where(
                        *[
                            getattr(self.table.c, pk_column) == pk_value
                            for pk_column, pk_value in primary_key_values.items()
                        ]
                    )
                    .values({column_name: new_value})
                )
                conn.execute(stmt)
                conn.commit()

            return 0
        except Exception as exc:
            logger.error(f"An error occurred: {exc}")
            sys.exit(-1)
        finally:
            conn.close()

    def delete(self, condition_dict: dict = None) -> int:
        """delete rows depending on condition_dict from the SQL table

        Args:
            condition_dict (dict, optional): condition to apply for columns of SQL table or index of SQL table. Defaults to None.

        Returns:
            int: returns 0, if successful.
        """

        try:
            if condition_dict is None:
                # Should we delete whole table data? or do nothing?
                # TODO
                return
            else:
                with self.engine.connect() as conn:
                    stmt_tc = None
                    for col, val in condition_dict.items():
                        stmt_c = getattr(self.table.c, col) == val
                        if stmt_tc == None:
                            stmt_tc = delete(self.table).where(stmt_c)
                        else:
                            stmt_tc = stmt_tc.where(stmt_c)
                    _ = conn.execute(
                        stmt_tc
                    )
                    conn.commit()
        except Exception as exc:
            logger.error(f"An error occurred during DELETE: {exc}")
            sys.exit(-1)
        finally:
            conn.close()
        return 0


    def connect(self, table_name: str) -> int:
        """connect to a SQL table already created

        Args:
            table_name (str): sql table name to load

        Returns:
            int: returns 0, if successful.
        """
        # TODO
        pass
    
    # --- New methods ---

    def select_columns(self, columns: list[str], condition_dict: dict = None, range_condition_dict: dict = None) -> pd.DataFrame:
        """
        Selects specified columns from the table, with optional filtering.
        Filters from condition_dict AND range_condition_dict are combined if both are provided.

        Args:
            columns (list[str]): A list of column names to select.
            condition_dict (dict, optional): Equality conditions. Defaults to None.
            range_condition_dict (dict, optional): Range conditions. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with selected columns and filtered rows.
                          Exits on database errors or configuration errors (e.g., invalid column name).
        """
        if not columns:
            logger.error("Columns list cannot be empty for select_columns.")
            sys.exit(-1) # Consistent with other error handling leading to exit

        df = pd.DataFrame()
        try:
            # Validate column names
            for col_name in columns:
                if not hasattr(self.table.c, col_name):
                    raise AttributeError(f"Column '{col_name}' not found in table '{self.table.name}'.")
            
            selected_column_objects = [getattr(self.table.c, col_name) for col_name in columns]
            stmt = select(*selected_column_objects)

            conditions_to_apply = []
            if condition_dict:
                for col, val in condition_dict.items():
                    if not hasattr(self.table.c, col):
                        raise AttributeError(f"Condition column '{col}' not found in table '{self.table.name}'.")
                    conditions_to_apply.append(getattr(self.table.c, col) == val)

            if range_condition_dict:
                for col, (lower, upper) in range_condition_dict.items():
                    if not hasattr(self.table.c, col):
                        raise AttributeError(f"Range condition column '{col}' not found in table '{self.table.name}'.")
                    conditions_to_apply.append(getattr(self.table.c, col).between(lower, upper))
            
            if conditions_to_apply:
                stmt = stmt.where(and_(*conditions_to_apply))

            with self.engine.connect() as conn: # 'conn' here is local to 'with' block
                df = pd.read_sql(stmt, conn)
        except AttributeError as ae: 
            logger.error(f"Configuration error in select_columns: {ae}")
            sys.exit(-1)
        except Exception as exc:
            logger.error(f"An error occurred during select_columns: {exc}")
            sys.exit(-1) 
        
        return df


    def get_data_by_ids(self, id_column_name: str, ids: list, select_columns: list[str]) -> dict:
        """
        Queries information by a list of IDs for a specific ID column and returns selected column values.

        Args:
            id_column_name (str): The name of the ID column to filter on (e.g., "annotation_data_id").
            ids (list): A list of ID values to query.
            select_columns (list[str]): A list of column names to retrieve for each ID.

        Returns:
            dict: A dictionary where keys are the IDs and values are dictionaries
                  containing the selected column data.
                  Example: {"id1":{"url":"url_val1","text":"text_val1"}, ...}
                  Returns an empty dictionary if no IDs are provided. Exits on errors.
        """
        if not ids:
            logger.info("No IDs provided to get_data_by_ids, returning empty dictionary.")
            return {}
        if not select_columns:
            logger.error("select_columns list cannot be empty for get_data_by_ids.")
            sys.exit(-1)

        output_dict = {}
        try:
            # Validate id_column_name
            if not hasattr(self.table.c, id_column_name):
                raise AttributeError(f"ID Column '{id_column_name}' not found in table '{self.table.name}'.")

            # Validate all select_columns
            for col_name in select_columns:
                if not hasattr(self.table.c, col_name):
                    raise AttributeError(f"Selected column '{col_name}' not found in table '{self.table.name}'.")

            columns_to_fetch_db_names = list(set([id_column_name] + select_columns))
            column_objects_to_fetch = [getattr(self.table.c, col) for col in columns_to_fetch_db_names]

            stmt = select(*column_objects_to_fetch).where(getattr(self.table.c, id_column_name).in_(ids))
            
            with self.engine.connect() as conn: # 'conn' here is local to 'with' block
                result_proxy = conn.execute(stmt)
                actual_fetched_column_names = list(result_proxy.keys())

                for row_tuple in result_proxy:
                    row_data_dict = {name: value for name, value in zip(actual_fetched_column_names, row_tuple)}
                    current_id = row_data_dict[id_column_name]
                    data_for_id = {col: row_data_dict[col] for col in select_columns}
                    output_dict[current_id] = data_for_id
        except AttributeError as ae:
            logger.error(f"Configuration error in get_data_by_ids: {ae}")
            sys.exit(-1)
        except Exception as exc:
            logger.error(f"An error occurred during get_data_by_ids: {exc}")
            sys.exit(-1)
        
        return output_dict