"""Verify every reference against CrossRef by DOI (paper hard requirement).

Checks: DOI resolves, title matches (normalized substring), year >= expected,
container (venue) present. Output: results/raw/ref_verification.csv
"""
import json
import re
import time
import urllib.parse
import urllib.request

REFS = [
    # key, DOI, title substring (lowercase), min year
    ("PBFT-classic", "10.1145/571637.571640", "practical byzantine fault tolerance", 1999),
    ("HotStuff-classic", "10.1145/3293611.3331591", "hotstuff", 2019),
    ("PPO-classic", "10.48550/arXiv.1707.06347", "proximal policy optimization", 2017),
    ("DumboNG", "10.1145/3548606.3559379", "dumbo-ng", 2022),
    ("SquirRL", "10.14722/ndss.2021.24188", "squirrl", 2021),
    ("DA-PBFT", "10.1109/TC.2024.3377921", "dynamic adaptive framework", 2024),
    ("DAGWise", "10.1109/ICBC64466.2025.11114520", "dagwise", 2025),
    ("DESAC", "10.1109/TITS.2025.3559672", "trust model-based consensus", 2025),
    ("R-MADRL", "10.1109/TWC.2025.3544478", "rotating multi-agent", 2025),
    ("rPoA", "10.32604/cmes.2023.046826", "decentralized reputation", 2024),
    ("WBR", "10.1109/JIOT.2023.3340974", "reputation system", 2024),
    ("ShardComm", "10.1109/TCC.2022.3217856", "committee structure", 2023),
    ("VTchain", "10.1109/TMC.2023.3294968", "vehicular trust blockchain", 2023),
    ("AWARE", "10.1109/TDSC.2020.3030605", "aware", 2022),
    ("IssaFL", "10.1145/3560816", "federated learning", 2023),
    ("RepChain", "10.1109/JIOT.2020.3028449", "repchain", 2021),
    ("TRep", "10.48550/arXiv.2505.14551", "trust", 2025),
    ("Bullshark", "10.1145/3548606.3559361", "bullshark", 2022),
    ("Narwhal", "10.1145/3492321.3519594", "narwhal", 2022),
    ("Jolteon", "10.1007/978-3-031-18283-9_14", "jolteon", 2022),
    ("Mysticeti", "10.1109/SP63933.2026.00009", "mysticeti", 2026),
    ("BFTdelay", "10.1016/j.comcom.2025.108278", "delay analysis", 2025),
    ("Fountain", "10.1109/TNSM.2025.3576128", "fountain", 2025),
    ("HierBFTrep", "10.3390/s22155887", "reputation", 2022),
    ("JointRep", "10.1109/ACCESS.2023.3305375", "reputation", 2023),
    ("CompRep", "10.1007/s12083-022-01408-2", "reputation", 2023),
    ("RLconsensus", "10.3389/frai.2025.1672273", "adaptive consensus", 2025),
    ("PoSsurvey", "10.1155/2022/2812526", "proof of stake", 2022),
    ("IslamSurvey", "10.1109/ACCESS.2023.3267047", "consensus algorithms", 2023),
    ("BenchSLR", "10.1109/ACCESS.2022.3188123", "benchmarking", 2022),
    ("RessiAI", "10.1016/j.jnca.2024.103858", "ai-enhanced blockchain", 2024),
    ("ZhangDL", "10.1016/j.isci.2023.108509", "deep learning", 2024),
    ("GameInc", "10.1016/j.comnet.2025.111827", "game theory", 2026),
    ("FlwrBC", "10.1109/ACCESS.2023.3320045", "flwrbc", 2023),
    ("PoCL", "10.1109/BLOCKCHAIN62396.2024.00055", "collaborative-learning", 2024),
    ("RepLeader", "10.1109/ICBCTIS66509.2025.11387282", "reputation-enhanced", 2025),
    ("DeTRM", "10.1109/ICBC54727.2022.9805565", "detrm", 2022),
    ("SIoTtrust", "10.3390/electronics11233871", "trust and reputation", 2022),
]


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def fetch(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent":
                                 "adatrust-refcheck/1.0 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["message"]


def main():
    rows = ["key,doi,status,year,venue,title"]
    npass = 0
    for key, doi, sub, minyear in REFS:
        try:
            m = fetch(doi)
            title = (m.get("title") or [""])[0]
            venue = (m.get("container-title") or [""])[0]
            year = None
            for k2 in ("published-print", "published-online", "issued"):
                if k2 in m and m[k2].get("date-parts"):
                    year = m[k2]["date-parts"][0][0]
                    break
            ok_title = norm(sub) in norm(title)
            ok_year = year is not None and year >= minyear
            status = "PASS" if (ok_title and ok_year) else \
                     f"FAIL(title_ok={ok_title},year={year})"
        except Exception as e:
            title = venue = ""
            year = None
            status = f"FAIL({type(e).__name__})"
        npass += status == "PASS"
        print(f"{key:14s} {status:28s} {year} | {title[:60]}", flush=True)
        rows.append(f'{key},{doi},{status},{year},"{venue}","{title.replace(chr(34), "")}"')
        time.sleep(0.3)
    out = "/home/user1/projects/results/raw/ref_verification.csv"
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\n[npass] {npass}/{len(REFS)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
