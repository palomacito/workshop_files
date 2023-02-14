"""
Connector object that handles and
returns SQL queries to the DWH.
Compatible with both Oracle and MySQL databases.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.types import String, Integer, Float, DateTime, Boolean

import pandas as pd
import numpy as np
import os
import unidecode
import re


def as_list(x):
    """Return input x as a list

    :param x: input object
    :return: x cast to list
    """
    try:
        return list(x)
    except:
        return [x]


def sanitize_string(string):
    """Removes accents and special characters from a string, replaces spaces with '_' and transforms it to CAPS

    :param string: input string
    :return: sanitized string
    """
    string = string.upper()
    string = unidecode.unidecode(string).replace(" ", "_")
    string = re.sub("[^a-zA-Z0-9_\n\.]", "", string)
    return string


def lena(x):
    """Returns the length of input x, if applicable, otherwise returns 0

    :param x: input variable
    :return: length of x or 0 if not applicable
    """
    try:
        return len(x)
    except:
        return 0


def adjust_dtypes(df, len_varchar=None):
    """
    Translate dtypes of a DataFrame into SQLAlchemy types.

    :param df: inout dataframe
    :param len_varchar: indicates the lenght of 'object' column elements
    :return: dictionnary containing the SQLAlchemy types for each field of the input dataframe
    """
    dtype = {}
    for c in df:
        if df[c].dtype == "object":
            s = df[c].dropna().apply(str)
            if len_varchar:
                maxlen = len_varchar
            elif len(s) == 0:
                maxlen = 1
            else:
                maxlen = int(s.apply(len).max() * 1.5) + 1
            dtype[c] = String(maxlen)
        elif df[c].dtype == "int64":
            dtype[c] = Integer
        elif df[c].dtype == "datetime64[ns]":
            dtype[c] = DateTime
        elif df[c].dtype == "float64":
            dtype[c] = Float
        elif df[c].dtype == "bool":
            dtype[c] = Boolean
        else:
            dtype[c] = df[c].dtype
    return dtype


class DWHConnector(object):
    """Helper class to handle interactions with the DWH.

    :param schema: DWH schema
    :param secrtets_path: path to secrets where credential files are kept
    :param engine: SQLAlchemy engine used to submit and retrieve SQL queries
    """

    def __init__(
        self,
        schema="WRK",
        secrets_path="/data/home/jdespraz/secrets",
        oracle_db=True,
        db_type=None,
    ):
        """Constructor method for the DWHConnector.
        NOTES: - the schema WRK allows to connect to the DTM schema but the opposite is not true.
               - files in the secrets path are assumed to be named as follows:
                   -> dwh_<schema_name>.txt for Oracle DB schemas
                   -> mysql.txt   for a Mysql custom DB

        :param schema: DWH schema to use (defaults to WRK)
        :param secrets_path: path to secrets where credential files are kept (defaults to '/data/home/jdespraz/secrets')
        :param oracle_db: Deprecated: boolean flag to specify database type (defaults to True for a PL/SQL Oracle database) -- kept for backwards compatibility -- use db_type instead
        :param db_type: optional string to specify the type of the target database connector
        """
        # set db_type default values depending on oracle_db flag
        # this is needed due to backward compatibility issues
        self.db_type = db_type
        if db_type is None:
            if oracle_db:
                self.db_type = "oracle"
            else:
                self.db_type = "mysql"
        self.schema = schema
        self.secrets_path = secrets_path
        self.engine = None
        self.connect()

    def get_schema(self):
        """Returns the schema used by the DWHConnector.

        :return: schema used by the DWHConnector
        """
        return self.schema

    def connect(self):
        """Initiates a connection to the DWH using the schema and credential provided by __init__() at object
        creation.

        :return: None
        """
        if self.db_type == "oracle":
            secrets = pd.read_csv(
                os.path.join(self.secrets_path, f"dwh_{self.schema.lower()}.txt"),
                sep=" ",
                header=None,
                index_col=0,
            )
            oracle_connection_string = (
                "oracle+cx_oracle://{username}:{password}@{tnsname}"
            )
            engine = create_engine(
                oracle_connection_string.format(
                    username=secrets.loc["username"].values[0],
                    password=secrets.loc["password"].values[0],
                    tnsname=secrets.loc["tnsname"].values[0],
                )
            )
            engine.execute(
                text("alter session set nls_date_format = 'DD.MM.YYYY HH24:MI:SS'")
            )
            self.engine = engine
            return None

        else:
            if self.db_type == "postgres":
                connection_prefix = "postgresql://"
                connection_suffix = ""
                fname = "postgres.txt"
            elif self.db_type == "mysql":
                connection_prefix = "mysql+pymysql://"
                connection_suffix = ""
                fname = "mysql.txt"
            elif self.db_type == "mssql":
                connection_prefix = "mssql+pyodbc://"
                connection_suffix = "?driver=ODBC+Driver+17+for+SQL+Server"
                fname = "mssql.txt"
            else:
                print(
                    f"WARNING: unknown database type '{self.db_type}': unable to connect"
                )
                return None

            secrets = pd.read_csv(
                os.path.join(self.secrets_path, fname),
                sep=" ",
                header=None,
                index_col=0,
            )
            username = secrets.loc["username"].values[0]
            password = secrets.loc["password"].values[0]
            host = secrets.loc["host"].values[0]
            port = int(secrets.loc["port"].values[0])
            database = secrets.loc["database"].values[0]
            engine = create_engine(
                f"{connection_prefix}{username}:{password}@{host}:{port}/{database}{connection_suffix}"
            )
            self.engine = engine
            return None

    def sql_query(self, query):
        """Executes a SQL query on the DWH and returns its output in a pandas dataframe.

        :param query: SQL query to execute on the DWH
        :return: pandas formatted SQL command output
        """
        if self.engine is not None:
            return pd.read_sql(query, self.engine)
        return None

    def sql_query_long(
        self, query_start, where_query, items, query_end="", maxitems=1000
    ):
        """Executes a SQL query on the DWH and returns its output in a pandas dataframe. This query allows the user
        to specify an arbitrary long list of parameters in a WHERE clause.
        Example: conn.sql_query_long("SELECT IPP FROM DTM_DSR.V_RFP_SEJ", "WHERE NUMERO_SEJOUR IN", df.index)

        :param query_start: initial SQL query (without the where clause)
        :param where_query: SQL where clause query
        :param items: list of elements contained in the where query
        :param query_end: optional final SQL query (after the where clause)
        :param maxitems: maximum number of elements supported in the SQL where query (defaults to 1000, i.e. system default)
        :return: pandas formatted SQL command output
        """
        if self.engine is None:
            return None

        items = as_list(items)  # to be safe

        # adding parenthesis to ensure correct behavior with multiple WHERE clauses
        idx = where_query.find("WHERE")
        if idx == -1:
            print("WARNING: missing WHERE keyword in the SQL query")
            return None
        idx += 5
        full_query = query_start + f" {where_query[:idx]} ( {where_query[idx:]}"

        nitems = len(items)
        for i in range(0, nitems, maxitems):
            sublist = items[i : i + maxitems]
            if len(sublist) == 1:
                # to avoid having queries ending with commas
                full_query += f" ('{sublist[0]}')"
            else:
                full_query += f" {tuple(sublist)}"
            if i + maxitems < nitems:
                full_query += "\n" + where_query.replace("WHERE", "OR")
        full_query += " ) " + query_end
        return pd.read_sql(full_query, self.engine)

    def execute(self, query):
        """Executes a query on the DWH and returns its output.

        :param query: SQL query to execute on the DWH
        :return: SQL command output
        """
        if self.engine is not None:
            self.engine.execute(query)
        return None

    def drop_table(self, table_name, verbose=False):
        """Drops a table on the DWH, after checking that it exists.

        :param table_name: name of the SQL table to be erased
        :param verbose: boolean flag (defaults to False), if True, prints information logs
        :return: None
        """

        if self.engine.dialect.has_table(self.engine, table_name):
            drop_table_stmt = "drop table " + table_name
            self.execute(drop_table_stmt)
            if verbose:
                print(f"Table {table_name} exists.")
                print(f"{table_name} has just been dropped.")
        else:
            if verbose:
                print(f"Table {table_name} does not exist.")
        return None

    def append_to_table(self, dataframe, table_name, dtypes=None):
        """Appends provided dataframe to an existing table on the DWH.
        All column names are sanitized and set to uppercase.
        :param dataframe: dataframe containing the data to be uploaded
        :param table_name: existing table name
        :param dtypes: parameters type (see pandas.to_sql() doc for complete info), allows the specification of the table datatypes.
        :return: None
        """
        # make a copy to avoid modifying the original dataframe object
        dataframe = dataframe.copy()
        # sanitize all column names using the dedicated function
        dataframe.columns = [sanitize_string(c) for c in dataframe.columns]
        dataframe.to_sql(
            table_name,
            self.engine,
            if_exists="append",
            chunksize=1000,
            index=False,
            dtype=dtypes,
        )
        return None

    def create_table_from_dataframe(
        self, dataframe, table_name, dtypes=None, autofix_varchar=False, verbose=False
    ):
        """Creates a new table on the DWH from the provided pandas dataframe.
        Thue usage of dtypes is particularly useful to avoid the creation of CLOB column types
        (and rather use a VARCHAR2 field). Note that the dictionnary does not have to match exactly
        all of the dataframe column names.
        All column names are sanitized and set to uppercase.
        Example:
        from sqlalchemy.types import String
        [...]
        dtype={'ACCOUCHEMENT': String(200), 'REMARQUES': String(4000)}

        :param dataframe: dataframe containing the data to be uploaded
        :param table_name: new table name
        :param dtypes: parameters type (see pandas.to_sql() doc for complete info), allows the specification of the table datatypes.
        :param verbose: boolean flag (defaults to False), if True, prints information logs
        :param autofix_varchar: boolean flag (defaults to False), if True, transforms columns of type 'object' to varchar of optimal length (avoids conversion to CLOB)
        :return: None
        """
        # make a copy to avoid modifying the original dataframe object
        dataframe = dataframe.copy()
        # sanitize all column names using the dedicated function
        dataframe.columns = [sanitize_string(c) for c in dataframe.columns]
        if autofix_varchar:
            obj_columns = dataframe.columns[[d == "object" for d in dataframe.dtypes]]
            lmax = dataframe[obj_columns].applymap(lena).max()
            lmax = np.maximum(lmax + 200, 3000)
            dtypes = {o: String(lmax[o]) for o in obj_columns}

        if dtypes is not None:
            dtypes = {sanitize_string(k): v for k, v in dtypes.items()}

        dataframe.to_sql(
            table_name, self.engine, dtype=dtypes, index=False, chunksize=1000
        )
        if verbose:
            print(f"{table_name.upper()} created in {self.get_schema()}")
        return None

    def create_table_from_list(self, table_name, column_name, data_list, verbose=False):
        """Creates a new table on the DWH
        Based on Yves' original python method.

        :param table_name: new table name
        :param column_name: name of the column that will contain the data stored in 'data_list'
        :param data_list: list of elements that will be inserted in the new table's 'column_name'
        :param verbose: boolean flag (defaults to False), if True, prints information logs
        :return: None
        """
        self.drop_table(table_name)

        query_c = f'create table "{table_name.upper()}" as '
        for d in data_list:
            query_c = (
                query_c
                + f"SELECT TO_NUMBER({d}) AS {column_name.upper()} FROM DUAL UNION "
            )

        query_c = query_c[:-6]  # remove the last UNION keyword
        self.execute(query_c)
        if verbose:
            print(f"{table_name.upper()} created in {self.get_schema()}")
        return None

    def encode(
        self,
        table,
        project_name,
        fields,
        offset_day=None,
        offset_hour=None,
        offset_minute=None,
        offset_second=None,
        suppl_fields=None,
        suffix="_C",
    ):
        """Encodes (pseudonymize) a table using the Oracle coding functions implemented by the data anaylsts.
        If the time offsets are not provided, a random offset is going to be computed based on the patient's
        IPP. It is a stronger coding mechanism than having a single offset for the entire cohort.

        :param table: name of the table to encode
        :param project_name: name of the project (used to compute the unique hash code)
        :param fields: dictionary of column names and type of data for the columns that need to be encoded
        :param offset_day: offset to apply on days
        :param offset_hour: offset to apply on hours
        :param offset_minute: offset to apply on minutes
        :param offset_second: offset to apply on seconds
        :param suppl_fields: additional fields not part of the dictionary "fields" that should be erased
        :param suffix: suffix added at the end of the original table name (defaults to '_C')
        :return: None
        """
        # compute automatic offsets based on patient's real IPP
        if offset_day is None:
            offset_day = "(MOD(IPP+51,7))*POWER(-1,MOD(IPP,2))"
        if offset_hour is None:
            offset_hour = "(MOD(IPP+92,24))*POWER(-1,MOD(IPP,2))"
        if offset_minute is None:
            offset_minute = "(MOD(IPP+14,60))*POWER(-1,MOD(IPP,2))"
        if offset_second is None:
            offset_second = "(MOD(IPP+71,60)+1)*POWER(-1,MOD(IPP,2))"

        tcoded = table + suffix

        # drop table if it already exists
        self.drop_table(tcoded)

        # create the command to create the new table, including all the new encoded fields
        cmd = f"CREATE TABLE {self.schema}_DSR.{tcoded} AS (\n SELECT s.*\n"
        for name, typ in fields.items():
            cmd += ",\n"
            if typ == "date":
                cmd += f"DTM_APP.ENCODE_DATE('{project_name}', 'DATE', s.{name}, {offset_day}, {offset_hour}, {offset_minute}, {offset_second}) AS {name}_CODE"
            elif typ == "birthday":
                cmd += f"DTM_APP.ENCODE_BIRTHDATE('{project_name}', 'DATE_NAISSANCE', s.{name}, TO_DATE('01.01.2020', 'DD.MM.YYYY'), 1) AS {name}_CODE"
            elif typ == "age":
                cmd += f"DTM_APP.ENCODE_AGE('{project_name}', 'DATE_NAISSANCE', s.DATE_NAISSANCE, s.DATE_ENTREE_SEJOUR, 1) AS {name}_CODE"
            elif typ == "ipp":
                cmd += f"DTM_APP.ENCODE_IPP('{project_name}', s.{name}, TO_DATE('19000101','yyyyMMdd')) AS {name}_CODE"
            elif typ == "stay":
                cmd += f"DTM_APP.ENCODE_ENCOUNTER('{project_name}', s.{name}) AS {name}_CODE"
            else:
                raise NameError(f"Unknown data type : {typ}")
        cmd += f"\nFROM {self.schema}_DSR.{table} s )"

        # execute the command
        self.execute(cmd)

        # erase original fields that have been coded
        to_drop = list(fields)
        if suppl_fields is not None:
            to_drop += suppl_fields
        cmd = f"ALTER TABLE {self.schema}_DSR.{tcoded} DROP ({', '.join(to_drop)})"
        self.execute(cmd)
        return None
