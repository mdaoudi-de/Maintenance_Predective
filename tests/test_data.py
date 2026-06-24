"""Tests unitaires du chargement des données et du split imposé.

Ces tests nécessitent les données (brutes ou en cache). Ils sont ignorés
proprement si elles sont absentes (utile en CI sans le dataset).
"""
import numpy as np
import pytest

from hydraulic_valve import config, data

_DATA_AVAILABLE = (
    (config.RAW_DATA_DIR / "PS2.txt").exists()
    or (config.PROCESSED_DIR / "PS2.npy").exists()
)
pytestmark = pytest.mark.skipif(not _DATA_AVAILABLE, reason="Données absentes")


def test_target_is_binary():
    y = data.get_target()
    assert set(np.unique(y)).issubset({0, 1})


def test_target_length():
    assert len(data.get_target()) == 2205


def test_train_test_split_sizes():
    train_idx, test_idx = data.train_test_indices()
    assert len(train_idx) == config.N_TRAIN == 2000
    assert len(test_idx) == 205
    # Disjoints et couvrant l'ensemble.
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
    assert len(train_idx) + len(test_idx) == 2205


def test_sensor_shapes():
    assert data.load_sensor("PS2").shape == (2205, 6000)
    assert data.load_sensor("FS1").shape == (2205, 600)
