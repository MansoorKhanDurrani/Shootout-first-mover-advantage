# Step 1 — Dataset & Business Question

## Dataset
**International Football Shootouts** (`shootouts.csv`) — 683 recorded penalty
shootout matches from international football, spanning 1967–2026.

Columns: `date`, `home_team`, `away_team`, `winner`, `first_shooter`

## Business Question
In a penalty shootout, does the team that shoots **first** have a higher
chance of winning?

This is a long-standing debate in football analytics. Since 2003, IFAB has
experimented with alternate shootout orders (e.g. "ABBA") specifically
because of concern that shooting first gives a psychological edge. If the
data shows a real, measurable first-shooter advantage, that has direct
implications for how captains and coaches approach the coin toss - a
genuine, real-world decision, not just a curiosity.

## Problem Type
**Binary classification** — the target is a category, not a number:
- `home_team_won = 1` → the home team won the shootout
- `home_team_won = 0` → the away team won the shootout

(If we were predicting something like "final score" or "number of goals",
that would be regression, since the target would be a continuous number.
Here the outcome is one of two discrete classes, so classification is the
right framing.)