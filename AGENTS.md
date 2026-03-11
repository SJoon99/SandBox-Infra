# Repository Guidelines

## Project Structure & Module Organization
This repository is GitOps infrastructure for Kubernetes, centered on Argo CD.
- `argocd/`: root Helm chart and bootstrap manifests (`bootstrap/root-app.yaml`).
- `argocd/apps/`: app groups by domain (`00-infra`, `01-storage`, `03-platform`, `06-DataX-WEKA`, etc.).
- `argocd/apps/*/values.yaml`: environment-specific Helm overrides.
- `argocd/apps/*/templates/`: app-of-apps templates.
- `network/`, `siteA/`, `siteB/`: network/site-specific configs.
- Root `gpu_node-*.yaml`: standalone infra test manifests.

Prefer app-local changes (for example, `argocd/apps/06-DataX-WEKA/...`) over cross-cutting edits.

## Build, Test, and Development Commands
Use manifest rendering/validation before commit:
- `helm lint ./argocd` - validate root chart syntax.
- `helm template sandbox ./argocd -f argocd/values.yaml` - render root app manifests.
- `helm template datax ./argocd/apps/06-DataX-WEKA -f argocd/apps/06-DataX-WEKA/values.yaml` - render a domain chart.
- `kubectl kustomize argocd/apps/00-infra/cilium/base` - validate Kustomize output.
- `kubectl apply --dry-run=client -f <file>.yaml` - quick schema/client validation.

## Coding Style & Naming Conventions
- YAML: 2-space indentation, no tabs.
- Keep keys grouped logically (`image`, `resources`, `auth`, `tls`).
- File naming: lowercase with hyphens (example: `ceph-obc-pydio.yaml`).
- Keep secrets out of Git; use external secret management or sealed/encrypted workflows.
- Minimize comments; keep only operationally useful notes.

## Testing Guidelines
No traditional unit-test suite in this repo; testing is manifest correctness and deploy safety.
- Render charts/Kustomize locally for every modified app.
- Check image references, namespace, and sync-wave ordering.
- For risky changes, include a rollback note in PR description.

## Commit & Pull Request Guidelines
Follow Conventional Commit style seen in history:
- `feat(scope): ...`
- `fix(scope): ...`
- `chore(scope): ...`

PRs should include:
- changed paths and target app(s),
- why the change is needed,
- verification commands run and key output,
- operational impact (sync order, downtime risk, rollback plan).

## Security & Configuration Tips
- Do not commit real credentials/tokens.
- Review external chart image registries/tags explicitly.
- Treat `values.yaml` edits as production-impacting; validate before Argo CD sync.
