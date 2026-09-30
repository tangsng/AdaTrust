"""Fix the 6 failed reference verifications:
- arXiv DOIs (PPO, TRep): verify via DataCite API
- SquirRL (NDSS): verify via doi.org resolution + title search on CrossRef
- AWARE, IssaFL, RepChain: find correct DOI via CrossRef title search
"""
import json
import re
import time
import urllib.parse
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent":
                                 "adatrust-refcheck/1.0 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r), r.status


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def crossref_search(title, rows=3):
    url = ("https://api.crossref.org/works?rows=%d&query.bibliographic=%s"
           % (rows, urllib.parse.quote(title)))
    data, _ = get(url)
    return data["message"]["items"]


def datacite(doi):
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi)
    data, _ = get(url)
    return data["data"]["attributes"]


print("== PPO (arXiv via DataCite) ==")
try:
    a = datacite("10.48550/arXiv.1707.06347")
    t = a["titles"][0]["title"]
    y = a.get("publicationYear")
    print("PASS" if "proximal policy optimization" in t.lower() and y == 2017
          else f"CHECK {y} {t}")
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:100])

print("== TRep games (arXiv via DataCite) ==")
try:
    a = datacite("10.48550/arXiv.2505.14551")
    t = a["titles"][0]["title"]
    y = a.get("publicationYear")
    print(f"year={y} title={t}")
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:100])

print("== SquirRL (CrossRef title search) ==")
try:
    for it in crossref_search("SquirRL Automating Attack Discovery Blockchain Incentive"):
        t = (it.get("title") or ["?"])[0]
        y = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        print(y, "|", t[:80], "|", it.get("DOI"), "|",
              (it.get("container-title") or ["?"])[0][:40])
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:100])

print("== AWARE (CrossRef title search) ==")
try:
    for it in crossref_search("AWARE Adaptive Wide-Area Replication fast resilient"):
        t = (it.get("title") or ["?"])[0]
        y = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        print(y, "|", t[:80], "|", it.get("DOI"), "|",
              (it.get("container-title") or ["?"])[0][:40])
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:100])

print("== IssaFL ACM CSUR (CrossRef title search) ==")
try:
    for it in crossref_search("Blockchain-based federated learning securing Internet of Things comprehensive survey"):
        t = (it.get("title") or ["?"])[0]
        y = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        print(y, "|", t[:80], "|", it.get("DOI"), "|",
              (it.get("container-title") or ["?"])[0][:40])
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:100])

print("== RepChain (CrossRef title search) ==")
try:
    for it in crossref_search("RepChain reputation-based secure fast high incentive blockchain sharding"):
        t = (it.get("title") or ["?"])[0]
        y = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        print(y, "|", t[:80], "|", it.get("DOI"), "|",
              (it.get("container-title") or ["?"])[0][:40])
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:100])
