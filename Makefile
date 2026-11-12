# Convenience targets.
#
#   make gate   # fmt + clippy + tests + tutorial pages in sync + documentation site
#   make help   # list targets
CARGO ?= $(shell command -v cargo 2>/dev/null || echo $(HOME)/.cargo/bin/cargo)
export PATH := $(HOME)/.cargo/bin:$(PATH)

# Rayon width for the test targets: libtest already runs tests in parallel.
TEST_THREADS ?= 8

DOCS_LOG ?= $(shell echo $${TMPDIR:-/tmp})/qntt-docs-site.log

.PHONY: help gate test clippy fmt fmt-fix tutorial-check docs-site

help:
	@echo "gate           fmt + clippy + test + tutorial-check + docs-site"
	@echo "test           cargo test --workspace"
	@echo "clippy         cargo clippy --workspace --all-targets -- -D warnings"
	@echo "fmt            cargo fmt --check"
	@echo "fmt-fix        cargo fmt"
	@echo "tutorial-check the tutorial pages match examples/tutorial"
	@echo "docs-site      build the documentation site (needs julia; skipped without it)"

gate: fmt clippy test tutorial-check docs-site
	@echo "✓ gate passed"

test:
	env RAYON_NUM_THREADS=$${RAYON_NUM_THREADS:-$(TEST_THREADS)} $(CARGO) test --workspace

clippy:
	$(CARGO) clippy --workspace --all-targets -- -D warnings

fmt:
	$(CARGO) fmt --check

fmt-fix:
	$(CARGO) fmt

tutorial-check:
	@python3 scripts/mktutorial.py --check

docs-site:
	@if ! command -v julia >/dev/null 2>&1; then \
	  echo "docs-site: SKIPPED — julia not on PATH (the site was not built)"; \
	else \
	  $(CARGO) build -p qntt-capi > $(DOCS_LOG) 2>&1 \
	    || { echo "docs-site: cargo build -p qntt-capi FAILED"; cat $(DOCS_LOG); exit 1; }; \
	  if [ ! -f docs/Manifest.toml ]; then \
	    echo "docs-site: instantiating docs/ (first run)"; \
	    julia --project=docs -e 'using Pkg; Pkg.develop(path = "julia"); Pkg.instantiate()' \
	      > $(DOCS_LOG) 2>&1 || { echo "docs-site: instantiate FAILED"; cat $(DOCS_LOG); exit 1; }; \
	  fi; \
	  julia --project=docs docs/make.jl > $(DOCS_LOG) 2>&1 \
	    || { echo "docs-site: makedocs FAILED"; cat $(DOCS_LOG); exit 1; }; \
	  echo "✓ docs-site passed"; \
	fi
