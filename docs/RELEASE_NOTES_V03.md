# v0.3

Adds portable snapshots and a native ordered-launcher replay worker; preserves the
v0.2 static checker code and entry points. Replay schema is
`radiance-kernel-replay/1`, ABI target 30. The Python replay module reports 0.3.0;
the retained static module still reports 0.2.0 deliberately.

New commands: `validate`, `run`, `candidate`, `expect`, `coverage` via `replay.sh`.
New opt-in integration: `integration/kbench-replay.patch`, CMake
`RAD_BUILD_REPLAY_SUITE`, kBench `--snapshot-dir`/`RAD_REPLAY_CAPTURE_DIR`.

Read the root README and `V03_REVIEW.md` for implemented versus unimplemented
production fidelity, the source-snapshot build blocker and GPU validation limits.
