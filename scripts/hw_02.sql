use baseball;

# dropping if any similar table by the name of intermediary table exists
drop table if exists game_batter_intm;

# creating intermediary table with game and batter_counts table
create table game_batter_intm as
select bt_C.game_id, bt_C.batter, gm.local_date, bt_C.hit, bt_C.atBat, year(gm.local_date) as game_year
from batter_counts bt_C inner join game gm on (bt_C.game_id = gm.game_id)
;

# creating index on the intermediary table's batter column
create index batter_idx  on game_batter_intm (batter);

# creating index on the intermediary table's local_date column
create index date_idx on game_batter_intm (local_date);

# drop any table with name historic_batt_avg
drop table if exists historic_batt_avg;

# creating the historic batting average table for all the batters
create table historic_batt_avg as
select batter, if( sum(atBat) = 0, 0, sum(hit) / sum(atBat)) as historic_batting_avg from game_batter_intm group by batter
;

# drop any table with name annual_batt_avg
drop table if exists annual_batt_avg;

# creating the annual batting average table for all the batters
create table annual_batt_avg as
select batter, year(local_date) as game_year, if( sum(atBat) = 0, 0, sum(hit) / sum(atBat)) as annual_batting_avg
from game_batter_intm group by batter, year(local_date)
;
select count(*) from annual_batt_avg;
# drop any table with name rolling_100_days_avg
drop table if exists rolling_100_days_avg;

# creating the rolling 100 days batting average table for all the batters
create table rolling_100_days_avg as
select gb_1.batter, gb_1.local_date, coalesce( sum(gb_2.hit), 0) as cumalative_hits, coalesce( sum(gb_2.atBat), 0) as cumalative_atBat, if( sum(gb_2.atBat) = 0 or sum(gb_2.atBat) is null, 0, sum(gb_2.hit) / sum(gb_2.atBat)) as rolling_100_days_avg
from game_batter_intm gb_1 left join game_batter_intm gb_2
        on ( datediff(gb_1.local_date, gb_2.local_date) <= 100 and datediff(gb_1.local_date, gb_2.local_date) > 0 and gb_1.batter = gb_2.batter)
group by gb_1.batter, gb_1.local_date
;
