import unittest
from types import SimpleNamespace

from carbonbench.runner import RunError, swap_used_bytes, verify_prompt_cache_disabled


class RunnerTests(unittest.TestCase):
    def test_parses_macos_swapusage(self):
        value = "vm.swapusage: total = 22528.00M  used = 21626.25M  free = 901.75M  (encrypted)"
        self.assertEqual(swap_used_bytes(value), int(21626.25 * 1024**2))

    def test_parses_gibibytes(self):
        self.assertEqual(swap_used_bytes("total = 4G used = 1.5G free = 2.5G"), int(1.5 * 1024**3))

    def test_missing_swapusage_is_unknown(self):
        self.assertIsNone(swap_used_bytes(None))
        self.assertIsNone(swap_used_bytes("unavailable"))

    def test_prompt_cache_preflight_passes_after_slot_scrub(self):
        client = FakeCacheClient([0, 0, 0])
        self.assertEqual(
            verify_prompt_cache_disabled(
                client, "model", num_ctx=4096, keep_alive="5m", seed=1
            ),
            0,
        )
        self.assertTrue(client.calls[1]["raw"])
        self.assertEqual(client.unloaded, ["model"])

    def test_prompt_cache_preflight_rejects_historical_restore(self):
        client = FakeCacheClient([0, 0, 24])
        with self.assertRaisesRegex(RunError, "restored 24 prompt tokens"):
            verify_prompt_cache_disabled(
                client, "model", num_ctx=4096, keep_alive="5m", seed=1
            )
        self.assertEqual(client.unloaded, ["model"])


class FakeCacheClient:
    def __init__(self, cached_counts):
        self.cached_counts = list(cached_counts)
        self.calls = []
        self.unloaded = []

    def generate(self, model, prompt, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        return SimpleNamespace(prompt_eval_cached_count=self.cached_counts.pop(0))

    def unload(self, model):
        self.unloaded.append(model)


if __name__ == "__main__":
    unittest.main()
