from collections import namedtuple

import numpy

GameState = namedtuple("GameStats", ["num_out", "player", "is_first"])


class GameModel:
    def __init__(self, num_players, num_spots):
        self.num_players = num_players
        self.num_spots = num_spots
