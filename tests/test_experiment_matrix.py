import unittest

from src.evaluate import expected_condition_count
from src.experiment_matrix import build_experiment_matrix


class ExperimentMatrixTests(unittest.TestCase):
    def test_expected_count(self):
        self.assertEqual(expected_condition_count(), 49)

    def test_condition_ids_are_unique(self):
        conditions = build_experiment_matrix()
        ids = [condition.condition_id for condition in conditions]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__": unittest.main()
