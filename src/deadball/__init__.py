"""Dead-ball detection: which frames are `D`, and how sure we are.

    python -m deadball.train_players SAMPLE_DATA_DIR      fit the player model on open games
    python -m deadball.run OUT_DIR                        label the competition match

Run from src/ with the repository root on PYTHONPATH (for games.py). See docs/dead_ball_rule.md.
"""
