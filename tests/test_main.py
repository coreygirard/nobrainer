import numpy as np
from hypothesis import assume, given
from hypothesis.strategies import floats, lists

from nobrainer.main import *


@given(lists(floats(min_value=-5, max_value=5)))
def test_sigmoid_property_basics(data):
    data = np.array(data)

    output = sigmoid(data)

    assert type(output) == type(
        np.array([1])
    )  # TODO: figure out a way to use isinstance
    for e in output:
        assert isinstance(e, float)
        assert 0 < e < 1


@given(floats(min_value=-5, max_value=5), floats(min_value=-5, max_value=5))
def test_sigmoid_property_increasing(i, j):
    assume(i != j)

    if i > j:
        i, j = j, i

    data = np.array([i, j])
    output = sigmoid(data)

    assert output[0] < output[1]


@given(floats(min_value=0, max_value=5), floats(min_value=0, max_value=5))
def test_sigmoid_property_diminishing_positive(i, j):
    assume(j > 0)

    a, b, c = i, i + j, i + j + j

    data = np.array([a, b, c])
    output = sigmoid(data)
    delta = [output[1] - output[0], output[2] - output[1]]

    assert delta[0] > delta[1]


@given(floats(min_value=-5, max_value=0), floats(min_value=0, max_value=5))
def test_sigmoid_property_diminishing_negative(i, j):
    assume(j > 0)

    a, b, c = i, i - j, i - j - j

    data = np.array([a, b, c])
    output = sigmoid(data)
    delta = [output[1] - output[0], output[2] - output[1]]

    assert delta[0] < delta[1]


def test_NeuronLayer():
    layer = NeuronLayer(13, 17)

    assert layer.synaptic_weights.shape == (17, 13)
    for row in layer.synaptic_weights:
        for weight in row:
            assert -1 <= weight <= 1
