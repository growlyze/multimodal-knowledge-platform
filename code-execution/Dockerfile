# Sandbox image for Open WebUI code execution (M1.2).
#
# The runtime base is upstream's Open Terminal *slim* image. Open Terminal is
# the code-execution engine Open WebUI talks to (Pyodide and Jupyter are
# legacy), the slim variant already runs the server as uid 1000, and it ships
# no compilers, no sudo, no Docker CLI and no pip.
#
# No pip in the final stage means third-party packages cannot be installed
# there — they are built in a matching interpreter and copied in, the same
# `--prefix=/install` trick upstream uses for its own dependencies. The
# builder tag MUST stay in lockstep with the runtime's interpreter (3.12.13)
# or the copied C extensions land on the wrong site-packages path.

# --------------------------------------------------------------- builder ---
FROM python:3.12.13-slim AS builder

# Needed only if a pin has no cp312 wheel; the current set is all-wheels, but
# leaving this out turns any future pin bump into a cryptic build failure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

# --------------------------------------------------------------- runtime ---
# Pinned 2026-10-06: `:slim` is a floating tag and had already drifted
# 0.13.0 -> 0.14.0 unnoticed before this was caught (found while verifying
# the Stop-button patch below needed re-checking against the running
# version). Bump deliberately by changing this digest, not by letting the
# tag move underneath the image - this is the exact failure class
# `apache/tika:latest-full` already taught this project once (never use
# floating image tags).
FROM ghcr.io/open-webui/open-terminal:slim@sha256:ba7a67129bbaf8618887eb6283d7e351275e2838c0751bbafebbce6b22938687

# Merges into /usr/local alongside the open-terminal server. pip is NOT
# dragged back in — only the requirements and their dependencies — so the
# sandbox still cannot install packages at runtime.
COPY --from=builder /install /usr/local

# data_tools is a plain package with no packaging metadata, so it goes on
# PYTHONPATH rather than being pip-installed. A missing directory fails the
# build here, which is the correct time to find out.
COPY data_tools /opt/pylib/data_tools

# The only thing in this stack that actually kills a stuck process by AGE
# (P45, P68). Invoked from a host cron via `docker exec`, not by anything
# in-container, so it needs no entrypoint wiring - it just has to be present
# and importable by a plain `python3` with no third-party deps (it uses
# urllib only). Complements, not replaces, the disconnect-triggered kill
# patched in below - that one fires on an explicit Stop, this one is the
# backstop for a runaway nobody is watching at all.
COPY sandbox_watchdog.py /opt/pylib/sandbox_watchdog.py
ENV PYTHONPATH=/opt/pylib

# Open WebUI's Stop button cancels a real backend asyncio.Task (confirmed by
# reading open_webui/tasks.py's stop_task()), which propagates into whatever
# that task is awaiting - including an in-flight call to this server. But
# Open Terminal shields its background process from exactly that
# cancellation, so Stop never reaches the subprocess (reported directly,
# M1.2 retest: `ps aux` after Stop still showed the command's workers at
# 92-97% CPU). Patches BOTH execute() and get_status() to kill the process
# on that cancellation instead of silently swallowing it. Runs at build
# time only - the runtime root filesystem is read_only (compose.yml). See
# patch_open_terminal.py for the full why and the exact-match guard that
# fails this build loudly if a future Open Terminal release reshapes the
# patched blocks (the base image is on a floating `:slim` tag and has
# already drifted 0.13.0 -> 0.14.0 unnoticed once).
COPY patch_open_terminal.py /tmp/patch_open_terminal.py
RUN python /tmp/patch_open_terminal.py && rm /tmp/patch_open_terminal.py

# Byte-compile ahead of time: __pycache__ cannot be written to a read-only
# /usr/local at runtime, so without this every session pays the cold-import
# cost of pandas again.
RUN python -m compileall -q /usr/local/lib/python3.12/site-packages /opt/pylib || true

# The root filesystem is read-only at runtime (see compose.yml and README.md). Every library
# that wants a cache or config directory is pointed at a tmpfs, or it raises
# on first import.
ENV MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/home/user \
    PYTHONDONTWRITEBYTECODE=1

# A process here gets a TTY but no one to type at it, so anything that pages
# blocks forever on input that can never arrive. Reported symptom: Gemini
# answered a data-analysis prompt by calling help() on a library function;
# pydoc.getpager() saw an interactive stdin, chose a terminal pager, and
# printed
#     WARNING: terminal is not fully functional
#     Press RETURN to continue
# then waited. The turn hung 10+ minutes (30+ on gemini-3.5-flash-lite) and
# the UI's Stop button did not cancel it, because the process was alive and
# blocked, not looping. Haiku and Sonnet never hit it - they don't call help().
# Reproduced and fixed here 2026-09-24: with TERM=dumb, getpager() returns
# plainpager and help(print) exits 0 immediately.
#
# This does NOT make the container immune to blocking on stdin in general -
# `input()` still hangs, verified - because that needs stdin redirected from
# /dev/null, which is Open Terminal's own spawn behaviour and not ours to set.
# Env vars close the reported path; the runaway-process watchdog (P45) is what
# covers the rest.
ENV TERM=dumb \
    PAGER=cat \
    GIT_PAGER=cat \
    PYTHONUNBUFFERED=1

# Upstream's entrypoint starts as root to chown $HOME and build an iptables
# egress firewall, then drops to uid 1000 via gosu. We want neither: egress is
# blocked by an `internal: true` Docker network instead, which is enforced by
# the kernel, cannot be undone from inside the container, and needs no
# capabilities. Skipping that script means the container is never root at any
# point, which is what makes `cap_drop: [ALL]` safe rather than fatal.
USER user

EXPOSE 8000

# /health is the one unauthenticated route, so the check needs no API key.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# tini reaps the processes a runaway command leaves behind.
ENTRYPOINT ["/usr/bin/tini", "--", "open-terminal"]
CMD ["run", "--host", "0.0.0.0", "--port", "8000"]
