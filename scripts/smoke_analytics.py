"""Analytics smoke test over whatever tracked matches exist."""
from vexga.analytics.features import robot_match_features, match_features
rm = robot_match_features()
print("robot-match rows:", len(rm))
print(rm.select(["match_id","slot","team","speed_mean","frac_offensive_half","frac_near_loader","frac_near_goal","endgame_in_own_park"]))
mf = match_features()
print(mf)
from vexga.analytics.scout import render_team
p = render_team("91915B")
print("scout report:", p)
