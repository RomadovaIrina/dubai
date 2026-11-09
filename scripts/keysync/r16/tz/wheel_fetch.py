#!/usr/bin/env python3
"""r16 TZ: prefetch the exact files of pinned requirement sets with aria2c (16 files x 8 connections) into one flat folder for
`uv pip install --find-links` (uv takes a find-links file over the index for the same version: measured, numpy 1.26.4 1.1 s vs ~90 s).
Why: on r16-q one TCP connection to PyPI gets 0.06-0.2 MB/s (pytorch / HF 3-5 MB/s), many connections 60-200 MB/s.
Picks per pin the wheel uv would pick (best tag for that Python, from PyPI + the cu128 torch index + the llama-cpp CPU index), else the sdist;
sha256 from the index is checked by aria2c. llama-cpp-python's SDIST also goes to --llama-out (the CUDA build must not see the CPU wheel).
    python3 wheel_fetch.py --out DIR --llama-out DIR2 --set 3.11:lock1.txt --set 3.10:spleeter.txt ..."""
import argparse, concurrent.futures as cf, hashlib, html, json, os, pathlib, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from packaging.tags import compatible_tags, cpython_tags, platform_tags
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

PT = "https://download.pytorch.org/whl/cu128"; LLAMA = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
PT_NAMES = {"torch", "torchvision", "torchaudio", "triton"}


def get(url, accept=None):
    for k in range(4):
        try:
            h = {"User-Agent": "dabai-wheel-fetch/1"}; h.update({"Accept": accept} if accept else {})
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
        except Exception: pass
        time.sleep(2 + 3 * k)
    return None


def simple_html(base, name):   # PEP 503 page -> [(filename, url, sha256 | None)]
    page = f"{base}/{canonicalize_name(name)}/"; b = get(page); out = []
    for href in re.findall(r'href="([^"]+)"', b.decode() if b else ""):
        u, _, frag = html.unescape(href).partition("#"); u = urllib.parse.urljoin(page, u)
        out.append((urllib.parse.unquote(u.rsplit("/", 1)[-1]), u, frag[7:] if frag.startswith("sha256=") else None))
    return out


def pypi(name, ver):
    b = get(f"https://pypi.org/pypi/{name}/{ver}/json")
    if b: return [(f["filename"], f["url"], f["digests"].get("sha256")) for f in json.loads(b)["urls"]]
    b = get(f"https://pypi.org/simple/{canonicalize_name(name)}/", "application/vnd.pypi.simple.v1+json")   # the version string differs (1.0 vs 1.0.0)
    return [(f["filename"], f["url"], f.get("hashes", {}).get("sha256")) for f in json.loads(b)["files"]] if b else []


def rank_for(pyver):
    v = tuple(int(x) for x in pyver.split(".")); plats = list(platform_tags())
    tags = list(cpython_tags(v, platforms=plats)) + list(compatible_tags(v, f"cp{v[0]}{v[1]}", platforms=plats))
    return {str(t): i for i, t in enumerate(tags)}


def pick(files, name, ver, rank):
    want, n0 = Version(ver), canonicalize_name(name); best = None; sdist = None
    for fn, u, sha in files:
        try:
            if fn.endswith(".whl"):
                n, v, _, tags = parse_wheel_filename(fn); r = min(rank.get(str(t), 10 ** 9) for t in tags)
                if n == n0 and v == want and r < 10 ** 9 and (best is None or r < best[0]): best = (r, fn, u, sha)
            elif fn.endswith((".tar.gz", ".zip")):
                n, v = parse_sdist_filename(fn)
                if n == n0 and v == want and sdist is None: sdist = (fn, u, sha)
        except Exception: continue
    return best, sdist


def pins(path):
    for ln in open(path):
        ln = ln.split("#", 1)[0].split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s,]+)$", ln)
        if m and " @ " not in ln: yield m.group(1), m.group(3)


def resolve(name, ver, pyver, rank):
    srcs = [("pypi", lambda: pypi(name, ver))]
    if "+" in ver or canonicalize_name(name) in PT_NAMES: srcs.insert(0, ("pt", lambda: simple_html(PT, name)))
    if canonicalize_name(name) == "llama-cpp-python": srcs.append(("llama", lambda: simple_html(LLAMA, name)))
    whl, sd = None, None
    for tag, f in srcs:   # the best-tagged wheel across the indexes (uv: unsafe-best-match), the first sdist
        w, s = pick(f(), name, ver, rank); sd = sd or s
        if w and (whl is None or w[0] < whl[0]): whl = w
    return (whl[1:] if whl else None), sd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--llama-out", required=True)
    ap.add_argument("--set", action="append", required=True, help="PYVER:requirements.txt"); a = ap.parse_args()
    out, lout = pathlib.Path(a.out), pathlib.Path(a.llama_out); out.mkdir(parents=True, exist_ok=True); lout.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for s in a.set:
        pyver, path = s.split(":", 1); rank = rank_for(pyver)
        for n, v in pins(path): jobs[(canonicalize_name(n), v, pyver)] = rank
    t0 = time.time(); files, missing = {}, []
    with cf.ThreadPoolExecutor(32) as ex:
        fut = {ex.submit(resolve, n, v, py, r): (n, v, py) for (n, v, py), r in jobs.items()}
        for f in cf.as_completed(fut):
            n, v, py = fut[f]; whl, sd = f.result(); chosen = whl or sd
            if chosen is None: missing.append(f"{n}=={v} (py{py})"); continue
            files[chosen[0]] = (chosen[1], chosen[2], out)
            if n == "llama-cpp-python" and sd: files["L:" + sd[0]] = (sd[1], sd[2], lout)
    print(f"resolved {len(jobs)} pins -> {len(files)} files in {time.time() - t0:.0f} s; missing {len(missing)}: {missing}", flush=True)
    lst = out / "aria2.list"
    with open(lst, "w") as fh:
        for fn, (u, sha, d) in sorted(files.items()):
            fh.write(f"{u}\n  out={fn.split(':', 1)[-1] if fn.startswith('L:') else fn}\n  dir={d}\n" + (f"  checksum=sha-256={sha}\n" if sha else ""))
    t1 = time.time()
    for k in range(3):
        rc = subprocess.call(["aria2c", "-i", str(lst), "-j", "16", "-x", "8", "-s", "8", "-k", "1M", "-c", "--check-integrity=true", "--file-allocation=none",
                              "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn", "--summary-interval=0", "--max-tries=8", "--retry-wait=3"])
        if rc == 0: break
        print(f"aria2c rc {rc}, retry {k + 1}", flush=True)
    bad = [fn for fn, (u, sha, d) in files.items() if sha and (not (d / fn.split(":", 1)[-1]).exists() or
           hashlib.sha256((d / fn.split(":", 1)[-1]).read_bytes()).hexdigest() != sha)]
    gb = sum((d / fn.split(":", 1)[-1]).stat().st_size for fn, (u, sha, d) in files.items() if (d / fn.split(":", 1)[-1]).exists()) / 1e9
    print(f"WHEEL_FETCH {'DONE' if not bad and not missing else 'INCOMPLETE'}: {len(files)} files {gb:.1f} GB in {time.time() - t1:.0f} s; sha bad {bad}; missing {missing}")
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
