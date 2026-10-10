"""CPU-count helpers that behave on shared clusters (SLURM).

`os.cpu_count()` reports every core on the machine, not the cores a scheduler gave this job;
using it oversubscribes a shared node. We prefer, in order:
    NUM_THREADS env var  >  SLURM_CPUS_PER_TASK  >  the process's CPU affinity mask  >  os.cpu_count()
"""
import os


def available_cpus() -> int:
    for var in ("NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        val = os.environ.get(var, "")
        if val.isdigit() and int(val) > 0:
            return int(val)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS / Windows
        return os.cpu_count() or 1


def configure_tf_threads(tf) -> None:
    """Limit TensorFlow's CPU thread pools to the cores this job actually has."""
    n = available_cpus()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, min(n, 4)))
    except RuntimeError:
        pass  # TensorFlow already initialised (e.g. inside a long test session); keep its settings
