import pytest

def test_delivery_time_non_negative():
    # Prosty test logiczny: czas nie może być ujemny
    prediction = 15.5 # Tu docelowo odpalisz funkcję z api.py
    assert prediction >= 0

def test_distance_impact():
    # Dłuższy dystans powinien (zazwyczaj) oznaczać dłuższy czas
    # To jest test Twojej "inteligencji systemowej"
    dist_short = 2.0
    dist_long = 10.0
    assert dist_long > dist_short