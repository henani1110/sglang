# Service ledger normalization notes

- The raw remote `service-ledger.csv` is preserved unchanged under `artifacts/remote_project/logs/` and in both downloaded project archives.
- `service-ledger-normalized.csv` pairs each actual GPU service start with its graceful signal and confirmed release. The 512-off start label in the raw ledger is ambiguous (`probe0-buffered`); its timestamp matches the `default-probe0-buffered512` runner status and resource files.
- The 04:09:36 preflight row was written before `nvidia-smi -q -d PCI` failed. No server PID, GPU allocation, or port listener was created. The later 04:11:07 `no_service` ledger entry records discovery/correction, not GPU release; the normalized row therefore records zero GPU seconds and leaves the release field empty.
- Total GPU time is the sum of the ten actual service start-to-release intervals: 1,031 seconds (17.183 minutes). Failed preflight time is excluded.
