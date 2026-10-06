# Build with: docker build -t invariantlab/package-candidate:py3.12 -f docker/package-candidate.Dockerfile docker
FROM mirror.gcr.io/library/python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d

RUN pip install --no-cache-dir numpy==2.4.6 pytest==9.1.1 pyyaml==6.0.3
