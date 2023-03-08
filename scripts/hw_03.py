import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("hw_03").getOrCreate()

database = "baseball"
user = "root"
password = 1997
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


def get_mariadb_data(sql, temp_table_name):
    # Create a data frame by reading data from Oracle via JDBC
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


def main():
    get_mariadb_data("select * from batter_counts", "batter_counts")
    get_mariadb_data("select * from game", "game")

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

    class customTransformer(Transformer):
        def _transform(self, game_batter_intm):
            df_game_batter_intm.createOrReplaceTempView("game_batter_intm")
            df_game_batter_intm.persist(StorageLevel.MEMORY_ONLY)

            df_rolling_100_days_avg = spark.sql(
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

    cust_transofrmer = customTransformer()
    df_rolling_100_days_avg = cust_transofrmer.transform(df_game_batter_intm)

    df_rolling_100_days_avg.createOrReplaceTempView("rolling_100_days_avg")
    df_rolling_100_days_avg.persist(StorageLevel.MEMORY_ONLY)

    spark.sql(
        "select batter,local_date,cumalative_atBat,cumalative_hits, rolling_100_days_avg from rolling_100_days_avg"
    ).show()


if __name__ == "__main__":
    sys.exit(main())
