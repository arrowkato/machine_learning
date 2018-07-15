delete from Point_Per_Year

insert into Point_Per_Year
(
 Year
,Player
,Age
,PTS
)
select
	eys.Year
	,eys.Player
	,eys.Age
	, sum(eys.PTS) as pts
from
	each_year_stats as eys
group by
	eys.Year
	, eys.Player

;
	

select * from Point_Per_Year where Player='Tim Duncan'



