"""Fetch full bibliographic records (authors, venue, volume/issue/pages, year)
for all 38 references from CrossRef (arXiv entries via DataCite).
Output: /home/user1/projects/results/raw/references_full.json
"""
import json
import time
import urllib.parse
import urllib.request

REFS = [
    ("R1", "10.1145/571637.571640", "crossref"),
    ("R2", "10.1145/3293611.3331591", "crossref"),
    ("R3", "10.48550/arXiv.1707.06347", "datacite"),
    ("R4", "10.1109/ACCESS.2023.3267047", "crossref"),
    ("R5", "10.1155/2022/2812526", "crossref"),
    ("R6", "10.1145/3492321.3519594", "crossref"),
    ("R7", "10.1145/3548606.3559361", "crossref"),
    ("R8", "10.1007/978-3-031-18283-9_14", "crossref"),
    ("R9", "10.1109/SP63933.2026.00009", "crossref"),
    ("R10", "10.1145/3548606.3559379", "crossref"),
    ("R11", "10.1016/j.comcom.2025.108278", "crossref"),
    ("R12", "10.1109/TNSM.2025.3576128", "crossref"),
    ("R13", "10.14722/ndss.2021.24188", "crossref"),
    ("R14", "10.1109/TDSC.2020.3030605", "crossref"),
    ("R15", "10.1109/TC.2024.3377921", "crossref"),
    ("R16", "10.1109/ICBC64466.2025.11114520", "crossref"),
    ("R17", "10.1109/TITS.2025.3559672", "crossref"),
    ("R18", "10.1109/TWC.2025.3544478", "crossref"),
    ("R19", "10.3389/frai.2025.1672273", "crossref"),
    ("R20", "10.32604/cmes.2023.046826", "crossref"),
    ("R21", "10.1109/JIOT.2023.3340974", "crossref"),
    ("R22", "10.1109/JIOT.2020.3028449", "crossref"),
    ("R23", "10.1109/TCC.2022.3217856", "crossref"),
    ("R24", "10.1109/TMC.2023.3294968", "crossref"),
    ("R25", "10.3390/s22155887", "crossref"),
    ("R26", "10.1109/ACCESS.2023.3305375", "crossref"),
    ("R27", "10.1007/s12083-022-01408-2", "crossref"),
    ("R28", "10.1109/ICBCTIS66509.2025.11387282", "crossref"),
    ("R29", "10.48550/arXiv.2505.14551", "datacite"),
    ("R30", "10.3390/electronics11233871", "crossref"),
    ("R31", "10.1109/ICBC54727.2022.9805565", "crossref"),
    ("R32", "10.1145/3560816", "crossref"),
    ("R33", "10.1016/j.jnca.2024.103858", "crossref"),
    ("R34", "10.1016/j.isci.2023.108509", "crossref"),
    ("R35", "10.1016/j.comnet.2025.111827", "crossref"),
    ("R36", "10.1109/ACCESS.2023.3320045", "crossref"),
    ("R37", "10.1109/BLOCKCHAIN62396.2024.00055", "crossref"),
    ("R38", "10.1109/ACCESS.2022.3188123", "crossref"),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent":
                                 "adatrust-refcheck/1.0 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def from_crossref(doi):
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    authors = []
    for a in m.get("author", []):
        fam, giv = a.get("family", ""), a.get("given", "")
        initials = " ".join(x[0] + "." for x in giv.replace("-", " ").split() if x)
        authors.append(f"{initials} {fam}".strip())
    year = None
    for k in ("published-print", "published-online", "issued"):
        if k in m and m[k].get("date-parts"):
            year = m[k]["date-parts"][0][0]
            break
    return dict(authors=authors,
                title=(m.get("title") or [""])[0],
                venue=(m.get("container-title") or [""])[0],
                volume=m.get("volume", ""), issue=m.get("issue", ""),
                pages=m.get("page", ""), year=year)


def from_datacite(doi):
    a = get("https://api.datacite.org/dois/" + urllib.parse.quote(doi))["data"]["attributes"]
    authors = []
    for c in a.get("creators", []):
        name = c.get("name", "")
        if "," in name:
            fam, giv = [x.strip() for x in name.split(",", 1)]
            initials = " ".join(x[0] + "." for x in giv.split() if x)
            authors.append(f"{initials} {fam}")
        else:
            authors.append(name)
    return dict(authors=authors, title=a["titles"][0]["title"],
                venue="arXiv", volume="", issue="", pages="",
                year=a.get("publicationYear"))


def ieee_authors(authors):
    if len(authors) > 6:
        return ", ".join(authors[:6]) + ", et al."
    return ", ".join(authors)


def main():
    out = {}
    for key, doi, src in REFS:
        try:
            rec = from_crossref(doi) if src == "crossref" else from_datacite(doi)
            rec["doi"] = doi
            rec["ieee_authors"] = ieee_authors(rec["authors"])
            out[key] = rec
            print(f"{key}: {rec['ieee_authors'][:60]} | {rec['venue'][:35]} | "
                  f"{rec['volume']}({rec['issue']}) {rec['pages']} {rec['year']}",
                  flush=True)
        except Exception as e:
            out[key] = dict(doi=doi, error=f"{type(e).__name__}: {e}")
            print(f"{key}: FAIL {e}", flush=True)
        time.sleep(0.3)
    path = "/home/user1/projects/results/raw/references_full.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("[saved]", path, flush=True)


if __name__ == "__main__":
    main()
