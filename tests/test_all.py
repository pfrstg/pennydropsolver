import pytest

import pennydropsolver


@pytest.mark.parametrize(
    "n,expected", [(2, 1), (3, 2), (4, 2), (5, 3), (7, 3), (8, 3), (9, 4)]
)
def test_bits_to_encode(n, expected):
    assert pennydropsolver.bits_to_encode(n) == expected


@pytest.mark.parametrize(
    "num_players,num_spots,expected_states,expected_max",
    [
        (2, 6, 25, 33),
        (3, 2, 13, 17),
    ],
)
def test_state_counts(num_players, num_spots, expected_states, expected_max):
    model = pennydropsolver.GameModel(num_players=num_players, num_spots=num_spots)
    assert model.num_states() == expected_states
    assert model.max_state_index() == expected_max


@pytest.mark.parametrize(
    "num_players,num_spots",
    [
        (2, 6),
        (3, 2),
        (4, 7),
    ],
)
def test_total_states(num_players, num_spots):
    model = pennydropsolver.GameModel(num_players=num_players, num_spots=num_spots)
    total_valid = 0
    for state_idx in range(model.max_state_index()):
        state = model.state_index_to_state(state_idx)
        if model.is_state_valid(state):
            total_valid += 1
            assert model.state_to_state_index(state) == state_idx, state


def test_simple():
    model = pennydropsolver.GameModel(num_players=2, num_spots=6)
