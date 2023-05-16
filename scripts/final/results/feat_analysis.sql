use baseball;

# analysing non-typical columns. Hours not doing well
select
    hour(time(local_date)) as hours
    , sum(if(if(b.home_runs > b.away_runs, 1, 0) = 1, 1, 0)) / count(*) as home_win_perc
    , count(*) as total_count
from game gm
    inner join boxscore b
        on gm.game_id = b.game_id
            and gm.type = 'R'
where away_runs != home_runs
group by hour(time(local_date))
order by home_win_perc desc
;


# Wind condn didn't do well either

select
    if( cast(regexp_replace(b.wind, '[^0-9]+', '') as int) > 22, 1, 0) as windy_or_not
    , sum(if(if(b.home_runs > b.away_runs, 1, 0) = 1, 1, 0)) / count(*) as home_win_perc
    , count(*) as total_count
from game gm
    inner join boxscore b
        on gm.game_id = b.game_id
            and gm.type = 'R'
            and away_runs != home_runs
group by windy_or_not
;

# I would'a guess the extreme weather condn to help home team! And it definitely does! But only got some 200 such records with extreme weather
select
    sum(if(if(b.home_runs > b.away_runs, 1, 0) = 1, 1, 0)) / count(*) as home_win_perc
from game gm
    inner join boxscore b
        on gm.game_id = b.game_id
            and gm.type = 'R'
            and away_runs != home_runs
            and cast(regexp_replace(b.temp, '[^0-9]+', '') as int) > 95 or cast(regexp_replace(b.temp, '[^0-9]+', '') as int) < 38
union all

select
    count(*) as total_count
from game gm
    inner join boxscore b
        on gm.game_id = b.game_id
            and gm.type = 'R'
            and away_runs != home_runs
            and cast(regexp_replace(b.temp, '[^0-9]+', '') as int) > 95 or cast(regexp_replace(b.temp, '[^0-9]+', '') as int) < 38
;



#I tried just the stadium column but that didn't do so well. So, I tried to find the win percentage for home team over the course of season
# still not an amazing feature but slightly better

create table stadium_home_team_win_perc as
select
    gm_res_lft.game_id
    , gm_res_lft.stadium_id
    , sum(gm_res_rght.home_team_wins) / count(*) as home_win_perc
from game_res gm_res_lft
    left outer join game_res gm_res_rght
        on (gm_res_lft.stadium_id = gm_res_rght.stadium_id and gm_res_lft.local_date > gm_res_rght.local_date)
group by
    gm_res_lft.game_id
    , gm_res_lft.stadium_id
order by
    gm_res_lft.game_id
    , gm_res_lft.stadium_id
;

# rain or dome slightly favors the home team with 58% chances of winning
select
    overcast
    , sum(if(if(b.home_runs > b.away_runs, 1, 0) = 1, 1, 0)) / count(*) as home_win_perc
    , count(*) as total_count
from game gm
    inner join boxscore b
        on gm.game_id = b.game_id
            and gm.type = 'R'
where away_runs != home_runs
group by overcast
;
