# Kapture-AI fork of Plano

This fork carries the Fikar patches on top of upstream [katanemo/plano](https://github.com/katanemo/plano).
The gateway that runs them is [Kapture-AI/plano-gateway](https://github.com/Kapture-AI/plano-gateway)
(configs and compose); the image pin lives in fikar-infra `versions.json` (`services.plano`).

## `main` is the Fikar build branch

Since KAN-606 (2026-10-21), `main` is upstream plus every Fikar patch:

| Ticket | Patch |
|---|---|
| KAN-434 | per-response metering, token usage labelled by organization |
| KAN-485 | `tracing.attribution_header` (schema and `crates/common/src/configuration.rs`) |
| KAN-529 | `POST /v1/embeddings` passthrough in brightstaff |
| KAN-553 | service, user and execution attribution; usage log lines (`FIKAR_USAGE_LOG_DIR`) |

New work branches off `main` as `KAN-<n>-<slug>` and merges back with a pull request. Every merge
builds `fikaracr.azurecr.io/plano:sha-<short>` and `:latest` (`.github/workflows/build-push.yml`).

Before `main` carried these patches, they lived on `fikar/metering-and-org` (head `405b1ad`, the image
production runs) and `main` built plain upstream, which rejected the production configs
(`'attribution_header' was unexpected`). That branch is merged into `main` and kept as it was; its
GHCR workflow (`fikar-image.yml`) is not on `main` and no longer used. Do not build from it.

## Config check

`build-push.yml` runs `tests/fikar/check-configs.sh` on the image it just built, before pushing:
the image's own `planoai.config_generator` must accept `tests/fikar/config-pinned.yaml` and
`config-routed.yaml`, copies of the gateway's production configs. When the gateway's configs gain a
field, copy them here again. To check an image already in the registry, run the workflow on `main`
with `image=fikaracr.azurecr.io/plano:<tag>`, or locally:

```bash
tests/fikar/check-configs.sh fikaracr.azurecr.io/plano:<tag>
```

## Syncing upstream

Merge `upstream/main` into a branch off `main` (merge commit, never a rebase or force push), resolve,
run `cargo fmt --all` and `cargo test -p brightstaff` in `crates/`, and open a pull request. The
config check on that pull request is the guard against upstream changing the schema under us.

Upstream's publishing workflows (Docker Hub, PyPI, release binaries) only run in `katanemo/plano`.
