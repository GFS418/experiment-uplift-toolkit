"""Download the Hillstrom e-mail experiment and verify its pinned SHA-256."""

from exptools.data import DEFAULT_PATH, HILLSTROM_SHA256, download

if __name__ == "__main__":
    path = download()
    print(f"OK {path.relative_to(DEFAULT_PATH.parents[2])} sha256={HILLSTROM_SHA256[:12]}...")
