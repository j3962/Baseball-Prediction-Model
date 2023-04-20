import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

# setting up database connection parameters

database = input("Enter database name: ")
user = input("Enter username: ")
password = input("Enter password: ")
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


# creating a dummy class for overriding the transformer class
class customTransformer(Transformer):
    def __init__(self, spark):
        self.spark = spark

    def _transform(self, df_game_batter_intm):
        df_game_batter_intm.createOrReplaceTempView("game_batter_intm")
        df_game_batter_intm.persist(StorageLevel.MEMORY_ONLY)

        df_rolling_100_days_avg = self.spark.sql(
            "select gb_1.batter,"
            " gb_1.local_date, "
            "coalesce( sum(gb_2.hit), 0) as cumalative_hits, "
            "coalesce( sum(gb_2.atBat), 0) as cumalative_atBat,"
            "round( if( sum(gb_2.atBat) = 0 or sum(gb_2.atBat) is null, 0, sum(gb_2.hit) / sum(gb_2.atBat)), 4) "
            "as rolling_100_days_avg "
            "from game_batter_intm gb_1 left join game_batter_intm gb_2 "
            "on ( datediff(gb_1.local_date, gb_2.local_date) <= 100 and "
            "datediff(gb_1.local_date, gb_2.local_date) > 0 "
            "and gb_1.batter = gb_2.batter) "
            "group by gb_1.batter, gb_1.local_date "
            "order by gb_1.batter, gb_1.local_date "
        )
        return df_rolling_100_days_avg


# taking data from the mariadb database
def get_mariadb_data(sql, temp_table_name, spark):
    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )
    df.createOrReplaceTempView(temp_table_name)
    df.persist(StorageLevel.MEMORY_ONLY)
    return


def main():
    # creating spark session and pulling data from mariadb
    spark = SparkSession.builder.appName("hw_03").getOrCreate()
    get_mariadb_data("select * from batter_counts", "batter_counts", spark)
    get_mariadb_data("select * from game", "game", spark)

    # creating intermediary table
    df_game_batter_intm = spark.sql(
        "select "
        "bt_C.game_id, "
        "bt_C.batter,"
        " gm.local_date,"
        " bt_C.hit, "
        "bt_C.atBat, "
        "year(gm.local_date) as game_year "
        "from batter_counts bt_C inner join game gm on (bt_C.game_id = gm.game_id)"
    )

    # creating an object of the dummy class

    cust_transofrmer = customTransformer(spark)
    df_rolling_100_days_avg = cust_transofrmer.transform(df_game_batter_intm)

    df_rolling_100_days_avg.createOrReplaceTempView("rolling_100_days_avg")
    df_rolling_100_days_avg.persist(StorageLevel.MEMORY_ONLY)

    spark.sql(
        "select batter,local_date,cumalative_atBat,cumalative_hits, rolling_100_days_avg from rolling_100_days_avg"
    ).show()


if __name__ == "__main__":
    sys.exit(main())