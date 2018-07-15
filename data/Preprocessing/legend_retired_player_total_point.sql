drop table legend_retired_player_total_point;

create table legend_retired_player_total_point
(
 id BIGINT AUTO_INCREMENT NOT NULL PRIMARY KEY
,Player VARCHAR(32)
,total_point float
)
;

insert into legend_retired_player_total_point
(
 Player 
,total_point 
)
select
	lp.Player
	, sum(lp.PTS) as total_point
from 
	legend_player_point_per_year as lp
group by
	lp.Player
order by
	lp.Player
;



