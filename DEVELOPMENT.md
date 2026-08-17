# Development

CUE uses the [Hatch] project manager with `uv` as its environment installer.
Hatch manages dependencies and runs tests, type checking, linting, and builds in
isolated environments.

[Hatch]: https://hatch.pypa.io/

## Testing

```bash
hatch test
```

Tests marked `model` download the published CUE checkpoint and are excluded by
default. Run them explicitly when `HF_TOKEN` is available:

```bash
hatch test -m model
```

## Type checking

```bash
hatch run types:check
```

## Formatting and linting

```bash
hatch fmt
hatch fmt --check
```

## Lockfile

```bash
uv lock --upgrade
```

## Packaging

```bash
hatch build
```

## Continuous integration

Testing, type checking, and formatting/linting are checked in
[CI](.github/workflows/ci.yml) on Python 3.12–3.14. Publishing a GitHub release
triggers the trusted PyPI publisher workflow.
