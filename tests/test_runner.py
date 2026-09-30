import unittest

from carbonbench.runner import swap_used_bytes


class RunnerTests(unittest.TestCase):
    def test_parses_macos_swapusage(self):
        value = "vm.swapusage: total = 22528.00M  used = 21626.25M  free = 901.75M  (encrypted)"
        self.assertEqual(swap_used_bytes(value), int(21626.25 * 1024**2))

    def test_parses_gibibytes(self):
        self.assertEqual(swap_used_bytes("total = 4G used = 1.5G free = 2.5G"), int(1.5 * 1024**3))

    def test_missing_swapusage_is_unknown(self):
        self.assertIsNone(swap_used_bytes(None))
        self.assertIsNone(swap_used_bytes("unavailable"))


if __name__ == "__main__":
    unittest.main()
