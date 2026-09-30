#!/usr/bin/env python3
"""ref_verify.py — batch-verify the 46 references in paper.md via Crossref.

Step 2.0 of nature-ref-verifier: DOI resolvability + title cross-check.
For each entry: request https://api.crossref.org/works/<DOI>, then fuzzy-match
the returned title against the cited title. Output a verdict per entry.
"""
import json
import re
import sys
import time
import urllib.request
import urllib.parse

PAPER = "paper/paper.md"
UA = {"User-Agent": "AdaTrust-ref-verify/1.0 (mailto:researcher@example.com)"}


def norm(s):
    """Normalize a title for fuzzy comparison (lowercase, drop punctuation)."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def title_sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    wa, wb = set(a.split()), set(b.split())
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def fetch_doi(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["message"]


def main():
    t = open(PAPER, encoding="utf-8").read()
    body = t[t.index("## References"):]
    refs = re.findall(r"^\[(\d+)\]\s+(.*)$", body, re.M)

    verdicts = {"ok": 0, "fix": 0, "warn": 0, "unverifiable": 0}
    print(f"{'#':>3} {'verdict':14} {'cited title':40} crossref title")
    print("-" * 110)
    results = []
    for num, txt in refs:
        doi_m = re.search(r"DOI:\s*(\S+)", txt)
        title_m = re.search(r'"([^"]+)"', txt)
        doi = doi_m.group(1).rstrip(".") if doi_m else None
        cited = title_m.group(1) if title_m else None
        if not doi:
            verdicts["unverifiable"] += 1
            results.append((num, "no-DOI", None, None, None))
            print(f"{num:>3} {'no DOI':14} {str(cited)[:40]:40} (manual check)")
            continue
        try:
            msg = fetch_doi(doi)
            ct = msg.get("title", [""])[0]
            sim = title_sim(cited, ct)
            if sim >= 0.7:
                v = "OK"
                verdicts["ok"] += 1
            elif sim >= 0.4:
                v = "CHECK(title)"
                verdicts["warn"] += 1
            else:
                v = "MISMATCH"
                verdicts["fix"] += 1
            results.append((num, v, doi, cited, ct))
            print(f"{num:>3} {v:14} {str(cited)[:40]:40} {str(ct)[:45]}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                verdicts["fix"] += 1
                results.append((num, "DOI-404", doi, cited, None))
                print(f"{num:>3} {'DOI 404':14} {str(cited)[:40]:40} {doi}")
            else:
                verdicts["unverifiable"] += 1
                print(f"{num:>3} {'HTTP '+str(e.code):14} {str(cited)[:40]:40}")
        except Exception as e:
            verdicts["unverifiable"] += 1
            print(f"{num:>3} {'ERR':14} {str(cited)[:40]:40} {str(e)[:40]}")
        time.sleep(0.3)  # be polite to Crossref

    print("\n==== 汇总 ====")
    print(f"OK: {verdicts['ok']} | 需修正(MISMATCH/404): {verdicts['fix']} | "
          f"需核对(title模糊): {verdicts['warn']} | 无法验证/无DOI: {verdicts['unverifiable']}")


if __name__ == "__main__":
    main()
