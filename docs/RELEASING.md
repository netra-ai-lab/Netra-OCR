# Releasing the Netra OCR app

The app ships as one Docker image, `ghcr.io/netra-ai-lab/netra-ocr`, for
Intel/AMD (`linux/amd64`) and ARM (`linux/arm64`: Apple Silicon Macs, ARM
servers) under the same tag. `.github/workflows/docker.yml` builds and
publishes it whenever a version tag is pushed.

## One-time setup

1. **Make the package public.** After the first release, open
   GitHub → netra-ai-lab → Packages → `netra-ocr` → Package settings →
   *Change visibility* → Public. Until then `docker pull` asks for a login.
   (The image's `org.opencontainers.image.source` label links the package to
   this repository automatically.)
2. **Docker Hub (optional, recommended).** Docker Desktop's search box only
   finds Docker Hub images, so this is what lets people install the app from
   the Docker Desktop window without a terminal.
   - Create the namespace (e.g. `netralab` — Docker Hub names can't contain
     hyphens) and a repository `netra-ocr`.
   - Create an access token with *Read & Write* scope.
   - In the GitHub repo: Settings → Secrets and variables → Actions:
     - variable `DOCKERHUB_IMAGE` = `netralab/netra-ocr`
     - secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`

   With the variable unset, the workflow publishes to GHCR only.

## Each release

1. Bump `version` in `pyproject.toml` (the app shows it in the footer and at
   `/v1/info`).
2. Commit and push to `master`.
3. Tag and push the tag:

   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```

4. Watch the *docker* workflow. Each architecture builds on its own runner
   and runs `docker/smoke_test.py` (every detector, both decoders, PDF input,
   every export), so a broken image fails here instead of on a user's machine.
   The final job publishes `1.1.0`, `1.1` and `latest`.
5. Check it from a clean machine:

   ```bash
   docker run --rm -p 8000:8000 ghcr.io/netra-ai-lab/netra-ocr:1.1.0
   ```

To test the pipeline without releasing, run the workflow by hand
(Actions → docker → *Run workflow*). It publishes only a `sha-<commit>` tag.

## Facts for the product page

Measured on the 1.1.0 image (i7-13700KF, 24 threads, Docker Desktop):

| | |
| :--- | :--- |
| Download | ~660 MB compressed; ~1.7 GB of layers on disk (Docker Desktop shows ~2.4 GB) |
| Memory | ~0.8 GB idle, 1.7 GB peak on a 10-page PDF — recommend 4 GB for Docker |
| Speed | 1.4–3.2 s per page; 24.6 s for a 10-page PDF. Slower CPUs take longer |
| GPU | Not needed (CPU-only image) |
| Network | None after the pull: every model is inside the image |
