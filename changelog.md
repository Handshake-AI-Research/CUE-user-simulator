[2026-08-17 13:43] Rename installable package to cue-simulator
- Reason: User asked for the installable package to be just `cue-simulator`.
- Description: Renamed the distribution from `cue-runtime` to `cue-simulator` in pyproject.toml, README badges/install extras, OpenAI extra hint, and uv.lock. Import package remains `cue`; CLIs remain `cue` and `cue-simulator`.

[2026-08-17 13:41] Expose cue-simulator CLI alias alongside cue
- Reason: User asked whether the tool could be called `cue` or `cue-simulator` rather than sounding like the package name.
- Description: Kept `cue` as the primary console script and added `cue-simulator` as the same entry point; README install section now states package name vs CLI names explicitly.
