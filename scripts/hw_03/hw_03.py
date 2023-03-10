# mariadb-example.py
from pyspark.sql import SparkSession

appName = "PySpark Example - MariaDB Example"
master = "local"
# Create Spark session
spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

sql = "select * from batter_counts"
database = "baseball"
user = "root"
password = "1997"
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

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

df.show()
