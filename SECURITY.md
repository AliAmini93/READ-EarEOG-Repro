# Security and privacy

If you find in this repository, or in its history, anything that could identify a person or a recording, any
recording data, credentials, or local file paths, do not copy or forward it and do not open a public issue about it.
Report it privately through GitHub's private vulnerability reporting (Security tab, "Report a vulnerability"),
describing the file and commit without quoting the sensitive content.

Before every push, maintainers run:

```bash
python scripts/release_scan.py
```

(and `--history` before any change of visibility), which fails on data and notebook files, any JSON file, files over
1 MB, absolute or home-relative paths, e-mail addresses and commit identities other than no-reply
addresses, credential-like strings, identifier-like codes, and terms of the non-public project records (stored as
salted hashes read from a maintainer-only file that is never committed; such hashes are obfuscation, not secrecy;
without that file the scan reports INCOMPLETE and exits with status 2).
It cannot recognise personal names, host names or free-text descriptions of people; an independent human review of
the complete history remains necessary.
