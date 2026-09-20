import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import independence_cutoff as cutoff


class CutoffTests(unittest.TestCase):
    def test_suffix_excludes_cutoff_iteration_and_keeps_final(self):
        x = np.arange(1, 151)[:, None]
        for p in range(20, 121, 10):
            actual = cutoff.suffix(x, p)
            self.assertEqual(actual.shape, (150-p, 1))
            self.assertEqual(actual[0, 0], p+1)
            self.assertEqual(actual[-1, 0], 150)
        np.testing.assert_array_equal(cutoff.suffix(x, 0), x)

    def test_every_test_receives_the_same_suffix(self):
        config = json.loads((Path(__file__).parent/'setup_independence_cutoff.json').read_text())
        x = np.arange(2*150*2).reshape(2, 150, 2)
        with patch.object(cutoff, '_CONTEXT', (config, x)), \
             patch.object(cutoff, 'ljung_box_test', return_value={'name':'lb'}) as lb, \
             patch.object(cutoff, 'autohsic_test', return_value={'name':'hsic'}) as hsic, \
             patch.object(cutoff, 'bds_test', return_value={'name':'bds'}) as bds:
            answer = cutoff.evaluate_task((10, 1))
            self.assertEqual(answer[:2], (10, 1))
            for call in [lb, hsic, bds]:
                np.testing.assert_array_equal(call.call_args.args[0], x[1, 120:])

    def test_rejects_too_short_windows(self):
        config = json.loads((Path(__file__).parent/'setup_independence_cutoff.json').read_text())
        data = {'costs': np.ones((2, 150, 2)), 'completed_runs': np.ones(2, bool),
                'setup': {'simulated_rewards': True}}
        cutoff.validate_config(config, data)
        config['cutoff_horizons'] = [146]
        with self.assertRaisesRegex(ValueError, 'fixed maximum lag'):
            cutoff.validate_config(config, data)


if __name__ == '__main__':
    unittest.main()
