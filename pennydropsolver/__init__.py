from collections import namedtuple

import numpy as np


def bits_to_encode(n):
    return 1 + int(np.log2(n - 1))


# The terminal sate will be all -1
GameState = namedtuple("GameStats", ["num_out", "player", "is_first"])


class GameModel:
    def __init__(self, num_players, num_spots):
        self.num_players = num_players
        self.num_spots = num_spots

        self._num_players_bits = bits_to_encode(self.num_players)
        self._num_spots_bits = bits_to_encode(self.num_spots)

    def num_states(self):
        return 1 + self.num_players * self.num_spots * 2

    def max_state_index(self):
        return 1 + (1 << self._num_players_bits) * (1 << self._num_spots_bits) * 2

    def is_terminal_state(self, state):
        return state.num_out == -1 and state.player == -1

    def is_state_valid(self, state):
        if self.is_terminal_state(state):
            return True
        return (
            state.num_out >= 0
            and state.num_out < self.num_spots
            and state.player >= 0
            and state.player < self.num_players
            and (state.is_first == True or state.is_first == False)
        )

    def state_to_state_index(self, state):
        if self.is_terminal_state(state):
            return 0
        return (
            1
            + (state.num_out << (self._num_players_bits + 1))
            + (state.player << 1)
            + (0 if state.is_first else 1)
        )

    def state_index_to_state(self, state_idx):
        if state_idx == 0:
            return GameState(-1, -1, True)
        state_idx -= 1
        is_first = False if (1 & state_idx) else True
        state_idx >>= 1
        player = state_idx & ~(~0 << self._num_players_bits)
        state_idx >>= self._num_players_bits
        num_out = state_idx
        return GameState(num_out=num_out, player=player, is_first=is_first)
