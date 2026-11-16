"""Every constant of the H/A possession model, in one place (docs/possession_HA_proposal.md §5).

Each one is either a football norm fixed in advance or tuned on the open-data reference set
(Metrica sample games, degraded). Nothing here was chosen by looking at the competition match or
its public ground truth. The three decoder constants (lambda, m, beta) are tuned by
possession.train on the reference set and stored with the evidence weights in ha_model.json,
the way src/deadball keeps its switch cost in player_model.json; the grids they are chosen from
are here.
"""

FPS = 25
PITCH = (105.3, 68.0)            # metres, BAR-ATH (games.py); the sample games pass --pitch

# ---------------------------------------------------------------- live spells (§4.0)
MIN_SPELL_S = 2.0                # norm: shorter live spells get one team, no switch

# ---------------------------------------------------------------- restart prior (§4.1)
TAG_LINK_S = 3.0                 # Smart Tagging timing looseness
ONSET_S = 3.0                    # norm: a restart is played within 3 s of the ball going live
FAST_BALL_MS = 4.0               # m/s; the dead-ball model's slow / played boundary
PRE_KICK_FRAMES = 5              # = LAG: the kick happened inside the speed lag before the fast frame

# ---------------------------------------------------------------- features (§4.2)
LAG = 5                          # frames for derived speeds (0.2 s), as the dead-ball scripts
MAX_INTERP_FRAMES = 12           # 0.48 s: ball gaps filled on a straight line (EDA norm)
WINDOWS_S = (2.0, 5.0, 10.0)     # centred rolling means
MIN_PLAYERS = 3                  # fewest outfield players a team shape needs
CARRY_DIST_M = 1.5               # = CONTROL_DIST_M: ball within a stride of the carrier
CARRY_DV = 2.0                   # m/s: ball moves with the carrier (poss_carries norm)
CARRY_MIN_S = 0.5                # shortest carry (poss_carries norm)
TOUCH_MIN_S = 0.2                # shortest contact for a pass chain (poss_pass_chains norm)
PASS_MIN_MS = 5.0                # a soft pass
PASS_MAX_S = 5.0                 # a long ball's flight with a bounce
MIN_CHAIN = 2                    # chains of >= 2 passes (precision >= 0.97 on Sample Game 3)

# ---------------------------------------------------------------- evidence model (§4.2)
CLIP = 3.0                       # |log-odds| per frame; fixed in advance
L2 = 1e-3                        # ridge on the standardised weights (as src/deadball)
NEWTON_STEPS = 50

# ---------------------------------------------------------------- decoder (§4.3), grids
LAMBDAS = (1.0, 2.5, 5.0, 10.0, 25.0, 50.0)     # switch cost, in summed log-odds
MIN_SEG_S = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)     # m, in seconds (converted to frames)
BETAS = (0.0, 5.0, 10.0, 25.0, 50.0)           # bonus on the restart team at the spell start

# ---------------------------------------------------------------- snapping (§4.4)
SNAP = False                     # off until the reference set says it helps (train.py reports)
SNAP_S = 3.0                     # search half-width; fixed in advance
CONTROL_DIST_M = 1.5             # norm + tracking error
CONTROL_SPEED_MS = 4.0           # norm
CONTROL_K = 5                    # frames (0.2 s)

# ---------------------------------------------------------------- reference set (Track B)
DEGRADE_SEED = 0                 # fixed: byte-identical reference sets
