# Contributing

Contributions that improve reproducibility, correctness, or clarity are welcome.

1. Open an issue describing the proposed change or experiment.
2. Keep benchmark changes reproducible: include hardware/software versions, exact commands, run counts, and raw outputs.
3. Never commit model weights, access tokens, local `.env` files, private datasets, or machine-specific absolute paths.
4. Run checks before opening a pull request:

   ```bash
   python -m pip install -e '.[dev]'
   ruff check .
   pytest
   ```

5. For performance claims, include run-level values and distinguish measured token throughput from event/chunk throughput.

By submitting a contribution, you agree that your contribution is provided under the repository's MIT license.
