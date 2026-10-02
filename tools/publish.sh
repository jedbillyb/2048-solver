#!/bin/sh
# Builds the Linux worker binaries on OCI and publishes them where supervisors fetch
# them (/g2048/files/ = /var/www/g2048/): g2048-linux-aarch64 native, and a static
# g2048-linux-x86_64 (musl, linked with rust-lld) for the laptop and the OptiPlex.
# Run from the checkout to publish, after bumping BUILD in src/dist.rs. Then
# `farm set version=N` and every Linux worker updates itself. The Windows exe is
# not built here.
set -e
cd "$(dirname "$0")/.."
CARGO=~/.cargo/bin/cargo
WWW=/var/www/g2048
want=$(sed -n 's/^pub const BUILD: u32 = \([0-9]*\);/\1/p' src/dist.rs)
[ -n "$want" ] || { echo "no BUILD in src/dist.rs"; exit 1; }
~/.cargo/bin/rustup target add x86_64-unknown-linux-musl >/dev/null 2>&1
nice -n 19 $CARGO build --release
CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld nice -n 19 $CARGO build --release --target x86_64-unknown-linux-musl
for pair in aarch64:target/release/g2048 x86_64:target/x86_64-unknown-linux-musl/release/g2048; do
  arch=${pair%%:*}; bin=${pair#*:}
  if [ "$arch" = "$(uname -m)" ]; then got=$($bin build); else got=$want; fi
  [ "$got" = "$want" ] || { echo "$bin reports build $got, expected $want"; exit 1; }
  # Copy then rename, so a supervisor never downloads a half-written file.
  sudo install -o "$(id -un)" -m 755 "$bin" "$WWW/.g2048-linux-$arch.tmp"
  sudo mv "$WWW/.g2048-linux-$arch.tmp" "$WWW/g2048-linux-$arch"
  echo "published g2048-linux-$arch build $want ($(stat -c %s "$WWW/g2048-linux-$arch") bytes)"
done
