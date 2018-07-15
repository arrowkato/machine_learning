drop table legend_retired_player_until32;

create table legend_retired_player_until32
(
 id BIGINT AUTO_INCREMENT NOT NULL PRIMARY KEY
, Player VARCHAR(32)
,18th float
,19th float
,20th float
,21th float
,22th float
,23th float
,24th float
,25th float
,26th float
,27th float
,28th float
,29th float
,30th float
,31th float
,32th float

)
;


insert into legend_retired_player_until32
(
 Player
,18th 
,19th 
,20th 
,21th 
,22th 
,23th 
,24th 
,25th 
,26th 
,27th 
,28th 
,29th 
,30th 
,31th 
,32th 


)
select
	pl.Player
	, max(case when pl.Age = 18 THEN PTS ELSE 0 END) as '18th'
	, max(case when pl.Age = 19 THEN PTS ELSE 0 END) as '19th'
	, max(case when pl.Age = 20 THEN PTS ELSE 0 END) as '20th'
	, max(case when pl.Age = 21 THEN PTS ELSE 0 END) as '21th'
	, max(case when pl.Age = 22 THEN PTS ELSE 0 END) as '22th'
	, max(case when pl.Age = 23 THEN PTS ELSE 0 END) as '23th'
	, max(case when pl.Age = 24 THEN PTS ELSE 0 END) as '24th'
	, max(case when pl.Age = 25 THEN PTS ELSE 0 END) as '25th'
	, max(case when pl.Age = 26 THEN PTS ELSE 0 END) as '26th'
	, max(case when pl.Age = 27 THEN PTS ELSE 0 END) as '27th'
	, max(case when pl.Age = 28 THEN PTS ELSE 0 END) as '28th'
	, max(case when pl.Age = 29 THEN PTS ELSE 0 END) as '29th'
	, max(case when pl.Age = 30 THEN PTS ELSE 0 END) as '30th'
	, max(case when pl.Age = 31 THEN PTS ELSE 0 END) as '31th'
	, max(case when pl.Age = 32 THEN PTS ELSE 0 END) as '32th'
from
	Legend_Player_Point_Per_Year as pl
group by
	pl.Player
order by
	pl.Player asc
;

