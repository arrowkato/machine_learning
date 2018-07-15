insert into Legend_Player_Point_Per_Year
(
  Year
  , Player
  , Age
  , PTS
)
select
	ppy.Year
	, ppy.Player
	, ppy.Age
	, ppy.PTS
from 
	Point_Per_Year as ppy
inner join
(
select
	Player as legend
from
	Point_Per_Year as eys
where
	eys.Player not in(
			select
				nest.Player
			from
				Point_Per_Year as nest
			where
				nest.Year = 2017
			group by
				nest.Player
	)
group by
	 eys.Player
having
	count(*) > 9
) legend_player
	on
		ppy.Player = legend_player.legend
order by
	ppy.Player asc
	, ppy.Year

;


