"""Array layouts shared by composites, the numba kernel and the interface. No group-label fields exist here."""
# player row columns (float64)
C_HEIGHT, C_REACH, C_VERT, C_QUICK, C_STR, C_TOUCH, C_VISION, C_TEMPER, C_END, C_MASS, \
    C_HAND, C_WING, C_VERTZ, C_OVR, C_AVAIL = range(15)
NF = 15

# coach row columns
CO_BIAS_RIM, CO_BIAS_MID, CO_BIAS_THREE, CO_PACE, CO_DEPTH, CO_FOUL = range(6)
NCO = 6

# box-score columns
ST_SEC, ST_PTS, ST_FGM, ST_FGA, ST_TPM, ST_TPA, ST_FTM, ST_FTA, ST_ORB, ST_DRB, ST_AST, \
    ST_STL, ST_BLK, ST_TOV, ST_PF, ST_RIM = range(16)
NSTAT = 16
STAT_NAMES = ("sec", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "ast",
              "stl", "blk", "tov", "pf", "rim")

# possession-log event codes; columns [period, clock, off_team, event, player_a, player_b, pts, flags]
EV_NSF, EV_TOV, EV_BLK, EV_MAKE, EV_MISS, EV_SFOUL = 1, 2, 3, 4, 5, 6
NLOG = 8

MAX_ROSTER = 13
