use baseball;

drop table if exists game_batter_intm;

create table if not exists game_batter_intm as
select bt_C.game_id, bt_C.batter, gm.local_date, bt_C.hit, bt_C.atBat, year(gm.local_date) as game_year
from batter_counts bt_C inner join game gm on (bt_C.game_id = gm.game_id)
;

create index batter_idx  on game_batter_intm (batter);

create index date_idx on game_batter_intm (local_date);

drop table if exists historic_batt_avg;

create table historic_batt_avg as
select batter, if( sum(atBat) = 0, Null, sum(hit) / sum(atBat)) as hist_batting_avg from batter_counts group by batter
;

drop table if exists annual_batt_avg;

create table annual_batt_avg as
select batter, year(local_date) as game_year, if( sum(atBat) = 0, Null, sum(hit) / sum(atBat)) as annual_avg
from game_batter_intm group by batter, year(local_date)
;

drop table if exists rolling_100_days_avg;

create table rolling_100_days_avg as
select gb_1.batter, gb_1.local_date, sum(gb_2.hit) as cum_hits, if( sum(gb_2.atBat) = 0, Null, sum(gb_2.hit) / sum(gb_2.atBat)) as roll_100_days_avg
from game_batter_intm gb_1 left join game_batter_intm gb_2
        on ( datediff(gb_1.local_date, gb_2.local_date) <= 100 and datediff(gb_1.local_date, gb_2.local_date) > 0 and gb_1.batter = gb_2.batter)
group by gb_1.batter, gb_1.local_date
;
