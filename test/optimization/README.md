# Optimization tests

Since they take a while, many optimization tests are marked as
`@pytest.mark.slow`. To run them, use the following command:

```bash
pytest -m slow
```

Some checks are skipped on mac/arm64 because some libraries are not available
(eg. nlopt).
