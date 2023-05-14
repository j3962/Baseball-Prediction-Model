#!/bin/bash

echo "inside the bash script"

sleep 35

echo "after the 30s zzzzleep"

if mysql -u root -p1997 -h jay_mariadb -e "select * from baseball.final_feat_stats limit 1;"
then
    echo "database is present just gonna create the feat stats tables"
    mysql -u root -p1997 -h jay_mariadb baseball < sql_scrpt_hw_05.sql
    echo "yayyy! feat stats table created"
else
    echo "gonna have to load the database. its not there"
    mysql -u root -p1997 -h jay_mariadb baseball < baseball.sql
    echo "database is ready! gonna run the feature script now"
    mysql -u root -p1997 -h jay_mariadb baseball < sql_scrpt_hw_05.sql
    echo "feature script done!"
fi

echo "running the ml code"

python3 main.py