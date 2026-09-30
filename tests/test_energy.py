import plistlib
import unittest

from carbonbench.energy import (
    EnergyError,
    PowerSample,
    parse_plist_document,
    parse_plist_stream,
    summarize_window,
)


def plist_sample(elapsed_ns=1_000_000_000, **processor):
    return plistlib.dumps({"elapsed_ns": elapsed_ns, "processor": processor})


class EnergyTests(unittest.TestCase):
    def test_parse_processor_power(self):
        sample = parse_plist_document(
            plist_sample(cpu_power=1000, gpu_power=2000.5, ane_power=3, combined_power=3003.5),
            end_monotonic_s=4.0,
            wall_time_s=10.0,
        )
        self.assertEqual(sample.elapsed_ns, 1_000_000_000)
        self.assertEqual(sample.cpu_mw, 1000)
        self.assertEqual(sample.soc_mw, 3003.5)
        self.assertEqual(sample.start_monotonic_s, 3.0)

    def test_missing_combined_sums_available_rails(self):
        sample = parse_plist_document(
            plist_sample(cpu_power=900, gpu_power=1100), end_monotonic_s=1.0
        )
        self.assertEqual(sample.soc_mw, 2000)
        self.assertIsNone(sample.ane_mw)

    def test_invalid_sample_is_preserved(self):
        data = plistlib.dumps(
            {"elapsed_ns": 1_000_000_000, "invalid": True, "processor": {"combined_power": 900}}
        )
        self.assertTrue(parse_plist_document(data).invalid)

    def test_nonpositive_elapsed_rejected(self):
        with self.assertRaises(EnergyError):
            parse_plist_document(plistlib.dumps({"elapsed_ns": 0, "combined_power": 1}))

    def test_nul_delimited_stream(self):
        data = plist_sample(combined_power=1000) + b"\x00" + plist_sample(combined_power=2000) + b"\x00"
        samples = parse_plist_stream(data)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0].monotonic_s, 1.0)
        self.assertAlmostEqual(samples[1].monotonic_s, 2.0)

    def test_exact_fractional_overlap_integration(self):
        sample = PowerSample(2.0, 2.0, 1_000_000_000, 1000, 2000, 0, 3000)
        result = summarize_window([sample], 1.5, 2.0, 500, 500)
        self.assertAlmostEqual(result.gross_soc_j, 1.5)
        self.assertAlmostEqual(result.incremental_soc_j, 1.25)
        self.assertAlmostEqual(result.sample_coverage, 1.0)

    def test_negative_incremental_energy_not_clamped(self):
        sample = PowerSample(1.0, 1.0, 1_000_000_000, 1000, None, None, 1000)
        result = summarize_window([sample], 0.0, 1.0, 500, 2000)
        self.assertAlmostEqual(result.incremental_soc_j, -1.0)

    def test_invalid_samples_do_not_contribute(self):
        sample = PowerSample(1.0, 1.0, 1_000_000_000, 1000, None, None, 1000, invalid=True)
        result = summarize_window([sample], 0.0, 1.0, 500, None)
        self.assertEqual(result.invalid_sample_count, 1)
        self.assertIsNone(result.gross_soc_j)
        self.assertEqual(result.sample_coverage, 0.0)


if __name__ == "__main__":
    unittest.main()

