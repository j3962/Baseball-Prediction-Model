use baseball;

drop table if exists game_res;

create table game_res as
select
    gm.game_id
    , gm.local_date
    , gm.away_l
    , gm.away_w
    , gm.home_w
    , gm.home_l
    , b.away_hits
    , b.away_runs
    , b.away_errors
    , b.home_hits
    , b.home_runs
    , b.home_errors
    , gm.home_team_id
    , gm.away_team_id
    , gm.home_pitcher
    , gm.away_pitcher
    , gm.stadium_id
    , b.wind
    , b.overcast
    , b.temp
    , if(b.away_runs > b.home_runs, 0, 1) as home_team_wins
from game gm
    inner join boxscore b
        on gm.game_id = b.game_id
            and gm.type = 'R'
            and b.away_runs != b.home_runs
;

drop table if exists pitcher_count_with_date;

create table pitcher_count_with_date as
select
    gm_res.local_date
    , gm_res.game_id
    , gm_res.away_l
    , gm_res.away_w
    , gm_res.home_w
    , gm_res.home_l
    , gm_res.away_hits
    , gm_res.away_runs
    , gm_res.away_errors
    , gm_res.home_hits
    , gm_res.home_runs
    , gm_res.home_errors
    , gm_res.home_team_wins
    , pc.pitcher
    , pc.team_id
    , pc.homeTeam
    , pc.awayTeam
    , pc.startingPitcher
    , pc.bullpenPitcher
    , pc.startingInning
    , pc.endingInning
    , pc.outsPlayed
    , pc.plateApperance
    , pc.atBat
    , pc.Hit
    , pc.toBase
    , pc.caughtStealing2B
    , pc.caughtStealing3B
    , pc.caughtStealingHome
    , pc.stolenBase2B
    , pc.stolenBase3B
    , pc.stolenBaseHome
    , pc.updatedDate
    , pc.Batter_Interference
    , pc.Bunt_Ground_Out
    , pc.Bunt_Groundout
    , pc.Bunt_Pop_Out
    , pc.Catcher_Interference
    , pc.Double_Play
    , pc.Fan_interference
    , pc.Field_Error
    , pc.Fielders_Choice
    , pc.Fielders_Choice_Out
    , pc.Fly_Out
    , pc.Flyout
    , pc.Force_Out
    , pc.Forceout
    , pc.Ground_Out
    , pc.Grounded_Into_DP
    , pc.Groundout
    , pc.Hit_By_Pitch
    , pc.Home_Run
    , pc.Intent_Walk
    , pc.Line_Out
    , pc.Lineout
    , pc.Pop_Out
    , pc.Runner_Out
    , pc.Sac_Bunt
    , pc.Sac_Fly
    , pc.Sac_Fly_DP
    , pc.Sacrifice_Bunt_DP
    , pc.Single
    , pc.Strikeout
    , pc.Triple
    , pc.Triple_Play
    , pc.Walk
    , pc.pitchesThrown
    , pc.DaysSinceLastPitch
from pitcher_counts pc
    inner join game_res gm_res on pc.game_id = gm_res.game_id
order by gm_res.local_date
    , pc.game_id
    , pc.team_id
;

create index if not exists pitcher_idx on pitcher_count_with_date (pitcher);

create index if not exists date_idx on pitcher_count_with_date (local_date);

drop table if exists pitcher_features;

create table pitcher_features as
select
    pc_left.team_id
    , pc_left.game_id
    , pc_left.local_date
    , pc_left.pitcher
    , if( sum(pc_right.Hit) = 0 or sum(pc_right.atBat) = 0, 0, sum(pc_right.Hit) / sum(nullif( pc_right.atBat, 0 ))) as batting_average_against
    , if( sum(pc_right.Walk) = 0 or sum(pc_right.Strikeout) = 0, 0, sum(pc_right.Strikeout) / sum(pc_right.walk)) as strikeout_to_walk_ratio
    , if( sum(pc_right.Strikeout) = 0 or sum(pc_right.pitchesThrown) = 0, 0, sum(pc_right.Strikeout) / sum(pc_right.pitchesThrown)) as pitch_count
    , if( sum(pc_right.Hit + pc_right.Walk) = 0 or sum(pc_right.pitchesThrown) = 0, 0, sum(pc_right.Hit + pc_right.Walk) / nullif(sum(pc_right.outsPlayed / 3), 0)) as whip
    , if( sum( 13 * pc_right.home_runs + 3 * (pc_right.Walk - pc_right.Hit_By_Pitch) - 2 * pc_right.Strikeout) = 0 or sum(pc_right.outsPlayed / 3) = 0, 0, sum( 13 * pc_right.home_runs + 3 * (pc_right.Walk - pc_right.Hit_By_Pitch) - 2 * pc_right.Strikeout) / sum((pc_right.outsPlayed / 3))) as fielding_indp_pitchin
    , if( sum(pc_right.Walk) * 9 = 0 or sum(pc_right.outsPlayed / 3) = 0, 0, sum( 27 * pc_right.Walk) / sum(pc_right.outsPlayed)) as bb_9
    , if( sum(pc_right.Hit) = 0 or sum(pc_right.outsPlayed) = 0, 0, sum( 3 * pc_right.Hit) / sum(pc_right.outsPlayed)) as hits_per_innings
    , if( sum(pc_right.outsPlayed) = 0 or count(pc_right.pitcher) = 0, 0, sum(pc_right.outsPlayed) / nullif(count(pc_right.pitcher), 0)) as ip_gs
    , if( sum(pc_right.home_w + pc_right.away_w) = 0 or sum(pc_right.home_w + pc_right.away_w + pc_right.home_l + pc_right.away_l) = 0, 0, sum(pc_right.home_w + pc_right.away_w) / nullif( sum(pc_right.home_w + pc_right.away_w + pc_right.home_l + pc_right.away_l), 0)) as overall_win_ratio
    , if( last_value(pc_right.home_w) = 0 or last_value(pc_right.home_w + pc_right.home_l) = 0, 0, last_value(pc_right.home_w) / nullif(last_value(pc_right.home_w) + last_value(pc_right.home_l), 0)) as home_win_ratio
    , if(last_value(pc_right.away_l) = 0 or last_value(pc_right.away_w + pc_right.away_l) = 0, 0, last_value(pc_right.away_w) / nullif(last_value(pc_right.away_w) + last_value(pc_right.away_l), 0)) as away_loss_ratio
    , if( sum(if(pc_right.startingPitcher = 1 and pc_right.home_team_wins = 1, 1, 0)) = 0 or count(pc_right.pitcher) = 0, 0, sum(if(pc_right.startingPitcher = 1 and pc_right.home_team_wins = 1, 1, 0)) / nullif(count(pc_right.pitcher), 0)) as starting_pitch_home_w
    , if( sum(if(pc_right.startingPitcher = 1 and pc_right.home_team_wins = 1, 1, 0)) = 0 or count(pc_right.pitcher) = 0, 0, sum(if(pc_right.startingPitcher = 1 and pc_right.home_team_wins = 0, 1, 0)) / nullif(count(pc_right.pitcher), 0)) as starting_pitch_away_l
from
    pitcher_count_with_date pc_left
    left outer join pitcher_count_with_date pc_right on pc_left.pitcher = pc_right.pitcher and datediff(pc_left.local_date, pc_right.local_date ) <= 100 and datediff(pc_left.local_date, pc_right.local_date ) > 0
group by
    pc_left.team_id
    , pc_left.game_id
    , pc_left.local_date
    , pc_left.pitcher
order by
    pc_left.team_id
    , pc_left.game_id
    , pc_left.pitcher
    , pc_left.local_date
;

drop table if exists avg_pitch_count;

create table avg_pitch_count as
select
    game_id
    , team_id
    , sum(pitch_count) / count(*) as avg_pitch_count
    , sum(whip) / count(*) as avg_whip
    , sum(batting_average_against) / count(*) as avg_batting_average_against
    , sum(bb_9) / count(*) as avg_bb_9
    , sum(strikeout_to_walk_ratio) / count(*) as avg_strikeout_to_walk
    , sum(ip_gs) / count(*) as avg_ip_gs
    , sum(fielding_indp_pitchin) / count(*) as avg_fip
from pitcher_features
group by
    game_id
    , team_id
order by
    game_id
    , team_id
;

create index if not exists team_idx on team_streak (team_id);

create index if not exists game_idx on team_streak (game_id);

drop table if exists team_streaks_intm;

create table team_streaks_intm as
select
    pc_d.game_id
    , pc_d.team_id
    , pc_d.local_date
    , last_value(ts_home.home_streak) as last_home_streak
    , last_value(ts_home.series_streak) as last_home_ser_streak
    , last_value(ts_away.away_streak) as last_away_streak
    , last_value(ts_away.series_streak) as last_away_ser_streak
from pitcher_count_with_date pc_d
    left join team_streak ts_home
        on pc_d.game_id > ts_home.game_id and pc_d.team_id = ts_home.team_id and pc_d.homeTeam = 1
    left join team_streak ts_away
        on pc_d.game_id > ts_away.game_id and pc_d.team_id = ts_away.team_id and pc_d.homeTeam = 0
group by
    pc_d.game_id
    , pc_d.team_id
    , pc_d.local_date
;

drop table if exists stadium_home_team_win_perc;

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

create index if not exists team_idx on team_streaks_intm (team_id);

create index if not exists game_idx on team_streaks_intm (game_id);

create index if not exists pitcher_idx on pitcher_features (pitcher);

create index if not exists game_idx on pitcher_features (game_id);

create index if not exists game_idx on avg_pitch_count (game_id);

create index if not exists team_idx on avg_pitch_count (team_id);

create index if not exists  game_idx on stadium_home_team_win_perc (game_id);

drop table if exists final_feat_stats;

create table final_feat_stats as
select
    gm_res.game_id
    , gm_res.home_team_id
    , gm_res.away_team_id
    , gm_res.home_pitcher
    , gm_res.away_pitcher
    , gm_res.local_date
    , pf_home.batting_average_against / nullif(pf_away.batting_average_against, 0) as batting_average_against_ratio
    , pf_home.strikeout_to_walk_ratio / nullif(pf_away.strikeout_to_walk_ratio, 0) as strikeout_to_walk_ratio
    , pf_home.pitch_count / nullif(pf_away.pitch_count, 0) as pitch_count_ratio
    , pf_home.whip / nullif(pf_away.whip, 0) as whip_ratio
    , pf_home.fielding_indp_pitchin / nullif(pf_away.fielding_indp_pitchin, 0) as fip_ratio
    , pf_home.bb_9 / nullif(pf_away.bb_9, 0) as bb_9_ratio
    , pf_home.ip_gs / nullif(pf_away.ip_gs, 0) as ip_gs_ratio
    , pf_home.overall_win_ratio / nullif(pf_away.overall_win_ratio, 0) as overall_win_ratio
    , pf_home.home_win_ratio / nullif(pf_away.away_loss_ratio, 0) as home_away_win_ratio
    , pf_home.starting_pitch_home_w / nullif(pf_away.starting_pitch_away_l, 0) as starting_pitch_home_w_ratio
    , ts_home.last_home_streak / nullif(ts_away.last_away_streak, 0) as home_away_strea_ratio
    , ts_home.last_home_ser_streak / nullif(ts_away.last_away_ser_streak, 0) as series_streak_ratio
    , gm_res.home_team_wins
from game_res gm_res
    left join pitcher_features pf_home
        on gm_res.local_date = pf_home.local_date and gm_res.home_pitcher = pf_home.pitcher
    left join pitcher_features pf_away
        on gm_res.local_date = pf_away.local_date and gm_res.away_pitcher = pf_away.pitcher
    left join team_streaks_intm ts_home
        on gm_res.game_id = ts_home.game_id and gm_res.home_team_id = ts_home.team_id
    left join team_streaks_intm ts_away
        on gm_res.game_id = ts_away.game_id and gm_res.away_team_id = ts_away.team_id
    left join avg_pitch_count apc_home
        on gm_res.game_id = apc_home.game_id and gm_res.home_team_id = apc_home.team_id
    left join avg_pitch_count apc_away
        on gm_res.game_id = apc_away.game_id and gm_res.away_team_id = apc_away.team_id
order by
    gm_res.game_id
    , gm_res.local_date
;

drop table if exists final_feat_stats_diff;

create table final_feat_stats_diff as
select
    gm_res.game_id
    , gm_res.home_team_id
    , gm_res.away_team_id
    , gm_res.home_pitcher
    , gm_res.away_pitcher
    , gm_res.local_date
    , (pf_home.batting_average_against - pf_away.batting_average_against) as batting_average_against_diff
    , (pf_home.strikeout_to_walk_ratio - pf_away.strikeout_to_walk_ratio) as strikeout_to_walk_diff
    , (pf_home.pitch_count - pf_away.pitch_count) as pitch_count_diff
    , (pf_home.whip - pf_away.whip) as whip_diff
    , (pf_home.fielding_indp_pitchin - pf_away.fielding_indp_pitchin) as fip_diff
    , (pf_home.bb_9 - pf_away.bb_9) as bb_9_diff
    , (pf_home.ip_gs - pf_away.ip_gs) as ip_gs_diff
    , (pf_home.overall_win_ratio - pf_away.overall_win_ratio) as overall_win_diff
    , (pf_home.home_win_ratio - pf_away.away_loss_ratio) as home_away_win_diff
    , (pf_home.starting_pitch_home_w - pf_away.starting_pitch_away_l) as starting_pitch_home_w_diff
    , (ts_home.last_home_streak - ts_away.last_away_streak) as home_away_strea_diff
    , (ts_home.last_home_ser_streak - ts_away.last_away_ser_streak) as series_streak_diff
    , if(std_win_perc.home_win_perc > .55, 1, 0) as std_home_win_perc
    , if(cast(regexp_replace(gm_res.temp, '[^0-9]+', '') as int) > 95 or cast(regexp_replace(gm_res.temp, '[^0-9]+', '') as int) < 38, 'Yes', 'No') as extreme_temp_event
    , if(gm_res.overcast = 'rain' or gm_res.overcast = 'dome', 'favorable', 'unfavorable') as fav_overcast
    , cast(hour(time(gm_res.local_date)) as varchar(5)) as match_start_hour
    , (apc_home.avg_batting_average_against - apc_away.avg_batting_average_against) as avg_batting_average_against_diff
    , (apc_home.avg_pitch_count - apc_away.avg_pitch_count) as avg_pitch_count_diff
    , (apc_home.avg_strikeout_to_walk - apc_away.avg_strikeout_to_walk) as avg_strikeout_to_walk_diff
    , (apc_home.avg_whip - apc_away.avg_whip) as avg_whip_diff
    , (apc_home.avg_fip - apc_away.avg_fip) as avg_fip_diff
    , (apc_home.avg_bb_9 - apc_away.avg_bb_9) as avg_bb_9_diff
    , (apc_home.avg_ip_gs - apc_away.avg_ip_gs) as avg_ip_gs_diff
    , gm_res.home_team_wins
from game_res gm_res
    left join pitcher_features pf_home
        on gm_res.local_date = pf_home.local_date and gm_res.home_pitcher = pf_home.pitcher
    left join pitcher_features pf_away
        on gm_res.local_date = pf_away.local_date and gm_res.away_pitcher = pf_away.pitcher
    left join team_streaks_intm ts_home
        on gm_res.game_id = ts_home.game_id and gm_res.home_team_id = ts_home.team_id
    left join team_streaks_intm ts_away
        on gm_res.game_id = ts_away.game_id and gm_res.away_team_id = ts_away.team_id
    left join avg_pitch_count apc_home
        on gm_res.game_id = apc_home.game_id and gm_res.home_team_id = apc_home.team_id
    left join avg_pitch_count apc_away
        on gm_res.game_id = apc_away.game_id and gm_res.away_team_id = apc_away.team_id
    left join stadium_home_team_win_perc std_win_perc
        on gm_res.game_id = std_win_perc.game_id
order by
    gm_res.game_id
    , gm_res.local_date
;
