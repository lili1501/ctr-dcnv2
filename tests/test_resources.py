import os

from ctr.resources import available_cpus


def test_num_threads_env_wins(monkeypatch):
    monkeypatch.setenv("NUM_THREADS", "3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "9")
    assert available_cpus() == 3


def test_slurm_allocation_used_when_no_override(monkeypatch):
    monkeypatch.delenv("NUM_THREADS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert available_cpus() == 8


def test_falls_back_to_affinity_and_ignores_garbage(monkeypatch):
    monkeypatch.setenv("NUM_THREADS", "abc")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "0")
    expected = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    assert available_cpus() == expected >= 1
