# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for measured intervals and allocation identity accounting."""

import importlib.util
from pathlib import Path
import json
import sqlite3
import unittest


class ProfileReportTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("profile_report.py")
        self.assertTrue(path.exists(), "profiling report implementation is missing")
        spec = importlib.util.spec_from_file_location("profile_report", path)
        self.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.report)

    def test_union_does_not_double_count_overlapping_gpu_work(self):
        self.assertEqual(self.report.interval_union([(10, 20), (12, 15), (18, 30), (40, 45)]), 25)
        self.assertEqual(self.report.interval_union([]), 0)

    def test_peak_matches_lifetimes_and_address_reuse(self):
        # timestamp, process, device, context, address, bytes, operation
        rows = [
            (1, 1, 0, 1, 100, 12, 0),
            (2, 1, 0, 1, 200, 8, 0),
            (3, 1, 0, 1, 100, 12, 1),
            (4, 1, 0, 1, 100, 16, 0),
            (5, 1, 0, 1, 200, 8, 1),
            (6, 1, 0, 1, 100, 16, 1),
        ]
        result = self.report.allocation_peak(rows, 0, 1)
        self.assertTrue(result["exact_for_traced_device_allocations"])
        self.assertEqual(result["peak_bytes"], 24)
        self.assertEqual(result["peak_timestamp_ns"], 4)
        self.assertEqual(result["final_live_bytes"], 0)
        self.assertEqual(result["allocation_count"], 3)

    def test_same_address_on_different_devices_is_not_an_alias(self):
        rows = [(1, 1, 0, 1, 100, 12, 0), (2, 1, 1, 1, 100, 8, 0)]
        result = self.report.allocation_peak(rows, 0, 1)
        self.assertEqual(result["peak_bytes"], 20)
        self.assertEqual(result["per_device"]["0"]["peak_bytes"], 12)
        self.assertEqual(result["per_device"]["1"]["peak_bytes"], 8)

    def test_incomplete_or_inconsistent_trace_cannot_claim_exact_peak(self):
        cases = [
            [(1, 1, 0, 1, 100, 12, 1)],  # allocation was outside capture
            [(1, 1, 0, 1, 100, 12, 0), (2, 1, 0, 1, 100, 8, 1)],
            [(1, 1, 0, 1, 100, 12, 0), (2, 1, 0, 1, 100, 12, 0)],
            [(1, 1, 0, 1, 100, 12, 9)],  # unknown operation
            [(1, 1, 0, 1, 100, 12, 0), (1, 1, 0, 1, 200, 8, 1)],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                result = self.report.allocation_peak(rows, 0, 1)
                self.assertFalse(result["exact_for_traced_device_allocations"])
                self.assertIsNone(result["peak_bytes"])
                self.assertTrue(result["anomalies"])

    def test_sqlite_summary_excludes_private_metadata_and_maps_cuda_device(self):
        # Minimal populated export schema; queries execute against real SQLite.
        with sqlite3.connect(":memory:") as connection:
            connection.executescript("""
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                INSERT INTO StringIds VALUES(1,'linear'),(2,'cuLaunchKernel');
                CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT);
                INSERT INTO META_DATA_EXPORT VALUES('EXPORT_SCHEMA_VERSION','3.28.0'),('PRIVATE_EXTRA','private sentinel');
                CREATE TABLE META_DATA_CAPTURE(name TEXT,value TEXT);
                INSERT INTO META_DATA_CAPTURE VALUES('PROCESS_0:ENVIRONMENT_VARIABLE','private sentinel');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,30,1,1,1,1,128,1,1);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,15,2);
                CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER,cudaId INTEGER,pid INTEGER,uuid TEXT);
                INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(1,0,123,'physical-gpu-one');
            """)
            result = self.report.summarize(connection)
        self.assertNotIn("private sentinel", json.dumps(result))
        self.assertEqual(result["kernel_count"], 1)
        self.assertEqual(result["kernel_span"]["api"][0]["total_ns"], 5)
        self.assertEqual(result["cuda_device_mapping"][0]["gpuId"], 1)
        self.assertEqual(result["cuda_device_mapping"][0]["cudaId"], 0)
        self.assertFalse(result["device_allocations"]["available"])


if __name__ == "__main__":
    unittest.main()
