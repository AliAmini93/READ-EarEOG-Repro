#!/usr/bin/env python3
"""Pre-push scan for material that must not be published.

    python scripts/release_scan.py              # files tracked by Git (or all files if Git is unavailable)
    python scripts/release_scan.py --history    # additionally every blob, commit message, identity and tag in history

Fails (exit 1) on:
  * data, archive or notebook files, and any JSON file (the repository ships no data);
  * files larger than 1 MB;
  * absolute or home-relative paths (POSIX and Windows);
  * e-mail addresses other than no-reply addresses, and commit identities that are not no-reply addresses;
  * credential-like strings (private keys, common token prefixes, password/token/secret assignments);
  * identifier-like strings (participant/subject ids, sub-01, P07-style codes);
  * terms of the non-public project records, compared as salted hashes read from a maintainer-only file outside the
    tracked tree (see _internal_terms). Such hashes are obfuscation, not secrecy (short terms can be recovered by
    guessing), which is why the list is never committed; without the file the scan reports INCOMPLETE and exits 2.

Limitations: personal names, host names and free-text descriptions of people cannot be detected reliably by pattern;
an independent human review of the complete history remains a release condition.
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SUFFIX = {'.npz', '.npy', '.edf', '.bdf', '.fif', '.xdf', '.asc', '.tsv', '.csv', '.mat', '.h5', '.hdf5', '.pkl',
               '.pickle', '.parquet', '.feather', '.sqlite', '.db', '.xls', '.xlsx', '.vhdr', '.vmrk', '.eeg', '.set',
               '.fdt', '.gdf', '.cnt', '.dat', '.zip', '.gz', '.tar', '.7z', '.docx', '.pptx', '.mp4', '.wav', '.ipynb'}
BINARY_SUFFIX = {'.png', '.pdf', '.jpg', '.jpeg', '.gif'}
NOREPLY = re.compile(r'(@users\.noreply\.github\.com|noreply@anthropic\.com)$', re.I)
PATTERNS = {
    'absolute path': re.compile(r'(?<![\w.])(/home/|/media/|/mnt/|/Users/|/tmp/|/var/|/srv/|/root/|/opt/|~/|[A-Za-z]:\\|\\\\[A-Za-z0-9.-]+\\)'),
    'e-mail address': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'credential': re.compile(r'(BEGIN [A-Z ]*PRIVATE KEY|gh[oprsu]_[A-Za-z0-9]{16,}|github_pat_|xox[abprs]-|'
                             r'AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|'
                             r'(?<![A-Za-z])(password|passwd|token|secret|api_key)(?![A-Za-z])\s*[:=]|'
                             r'Bearer\s+[A-Za-z0-9._-]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})', re.I),
    'identifier': re.compile(r'((participant|subject)[_ -]?id(?![A-Za-z])|(?<![A-Za-z0-9])sub[-_]\d+|'
                             r'(?<![A-Za-z0-9])(?:VP|P|S)\d{2,3}(?![0-9]))', re.I),
}
SALT = 'read-eareog-release-scan-v1:'
def _internal_terms():
    """Hashes of non-public record names, read from a maintainer-only file that is never committed
    (default: .git/info/release-scan-terms; or the file named by READ_EAREOG_SCAN_TERMS). Returns None if absent."""
    import os
    path = os.environ.get('READ_EAREOG_SCAN_TERMS') or str(ROOT / '.git' / 'info' / 'release-scan-terms')
    try:
        return {ln.strip() for ln in open(path, encoding='utf-8') if ln.strip()}
    except OSError:
        return None


INTERNAL = _internal_terms()
SELF = 'scripts/release_scan.py'


def git(*args, binary=False):
    out = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode('utf-8', errors='replace')


def scan_text(name, text, problems, path=''):
    for label, rx in PATTERNS.items():
        if path == SELF and label != 'e-mail address':
            continue                                   # the scanner's own patterns
        for m in rx.finditer(text):
            if label == 'e-mail address' and NOREPLY.search(m.group(0)):
                continue
            problems.append(f'{name}: {label} ({m.group(0)[:40]!r})')
            break
    toks = set(re.findall(r'[a-z0-9_]+', text.lower()))
    toks |= {part for tok in toks for part in tok.split('_') if part}      # also the parts of snake_case names
    for tok in sorted(toks):
        if INTERNAL and hashlib.sha256((SALT + tok).encode()).hexdigest()[:16] in INTERNAL:
            problems.append(f'{name}: term of the non-public project records ({tok!r})')


def check_file(name, path, data: bytes, problems):
    suffix = Path(path).suffix.lower()
    if suffix in DATA_SUFFIX:
        problems.append(f'{name}: data, archive or notebook file type')
    if suffix == '.json':
        problems.append(f'{name}: JSON file (data or per-recording output?)')
    if len(data) > 1_000_000:
        problems.append(f'{name}: larger than 1 MB')
    if suffix not in BINARY_SUFFIX:
        scan_text(name, data.decode('utf-8', errors='replace'), problems, path)


def main():
    problems = []
    try:
        tracked = [f for f in git('ls-files').splitlines() if f]
    except (OSError, subprocess.CalledProcessError):
        tracked = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
    for rel in tracked:
        check_file(rel, rel, (ROOT / rel).read_bytes(), problems)
    if '--history' in sys.argv:
        seen = set()
        for line in git('rev-list', '--objects', '--all').splitlines():
            sha, _, name = line.partition(' ')
            if not name or sha in seen or git('cat-file', '-t', sha).strip() != 'blob':
                continue
            seen.add(sha)
            check_file(f'history {name} ({sha[:8]})', name, git('cat-file', '-p', sha, binary=True), problems)
        for rev in git('rev-list', '--all').splitlines():
            scan_text(f'commit {rev[:8]} message', git('log', '-1', '--format=%B', rev), problems)
            for ident in git('log', '-1', '--format=%ae%n%ce', rev).split():
                if not NOREPLY.search(ident):
                    problems.append(f'commit {rev[:8]}: identity is not a no-reply address')
        for tag in git('tag', '-l').split():
            scan_text(f'tag {tag}', git('tag', '-l', '--format=%(contents)', tag), problems)
            for ident in git('tag', '-l', '--format=%(taggeremail)', tag).split():
                if ident.strip('<>') and not NOREPLY.search(ident.strip('<>')):
                    problems.append(f'tag {tag}: tagger identity is not a no-reply address')
    for p in problems:
        print('FAIL', p)
    if INTERNAL is None:
        print(f'release scan: {len(problems)} problem(s); INCOMPLETE - no local term list, so names of non-public '
              'records were not checked (do not treat this as a clean result)')
        return 1 if problems else 2
    print(f'release scan: {len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
