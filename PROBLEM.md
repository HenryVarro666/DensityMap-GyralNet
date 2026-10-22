# Research question

> **Given the same cortical surface input, can completing broken gyral skeletons make the network's connectivity more consistent without adding links that the geometry does not support?**

**Why it matters:** Downstream graph analysis depends on stable nodes and edges. A broken skeleton or a wrong link changes the network's structure even when the visualisation still looks plausible.

**Central tension:** Repairing breaks raises connectivity, but it can also join structures that are not actually adjacent. A more complete graph is not necessarily a more accurate one.

**Current evidence and gap:** [README.md](README.md) describes surface reading, isolated-point handling, skeleton creation and missing-link repair. That documents how the method runs, but it does not establish independent evidence that the repaired graph is more accurate than the unrepaired one.

**Smallest decisive experiment:** Fix the surfaces and parameter ranges and compare no repair with the current repair step. Under known, deliberately deleted edges and controlled mesh perturbations, measure recovery rate, false-link rate and topological stability, and have an independent sample reviewed blind. Keep the data used for tuning apart from the data used for the final evaluation. If the improvement shows up only as more edges, or depends on a single mesh, narrow the claim about the method.

This states a question to be tested; it adds no new measurements.

中文版：[PROBLEM.zh.md](PROBLEM.zh.md)
