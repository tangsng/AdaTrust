"""Discover + verify recent (2021+) references via CrossRef API.

For each candidate topic query, fetch top matches from api.crossref.org,
filter year >= 2021, and print structured candidates with DOI.
All selections in the paper come from this script's output - nothing invented.
"""
import json
import sys
import time
import urllib.parse
import urllib.request

QUERIES = [
    "Bullshark DAG BFT protocols made practical",
    "Narwhal and Tusk DAG-based mempool efficient BFT consensus",
    "Jolteon and Ditto network-adaptive efficient consensus asynchronous fallback",
    "Mysticeti low-latency DAG consensus fast commit path",
    "blockchain sharding survey scalability",
    "trust management blockchain survey reputation",
    "reputation based Byzantine fault tolerance consensus",
    "reinforcement learning consensus protocol optimization blockchain",
    "adaptive Byzantine fault tolerance consensus dynamic",
    "Ethereum proof of stake finality analysis",
    "Hyperledger Fabric performance evaluation throughput latency",
    "committee selection blockchain consensus random",
    "deep reinforcement learning blockchain edge computing resource",
    "proof of stake consensus survey security",
    "game theory blockchain consensus incentive mechanism",
    "dynamic committee election blockchain consensus",
    "blockchain consensus machine learning adaptive security 2024",
    "Byzantine fault tolerance IoT blockchain lightweight 2024",
    "federated learning blockchain consensus incentive 2023",
    "Markov decision process blockchain parameter optimization",
]


def query_crossref(q, rows=3):
    url = ("https://api.crossref.org/works?rows=%d&query.bibliographic=%s"
           % (rows, urllib.parse.quote(q)))
    req = urllib.request.Request(url, headers={"User-Agent":
                                 "adatrust-refcheck/1.0 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def main():
    out = []
    for q in QUERIES:
        try:
            data = query_crossref(q)
        except Exception as e:
            print(f"[fail] {q}: {e}", flush=True)
            continue
        items = data["message"]["items"]
        print(f"\n### {q}", flush=True)
        for it in items:
            year = None
            for key in ("published-print", "published-online", "issued"):
                if key in it and it[key].get("date-parts"):
                    year = it[key]["date-parts"][0][0]
                    break
            title = (it.get("title") or ["?"])[0]
            venue = (it.get("container-title") or ["?"])[0]
            doi = it.get("DOI", "?")
            typ = it.get("type", "?")
            flag = "OK" if (year and year >= 2021) else "OLD"
            print(f"[{flag}] {year} | {typ} | {title[:90]} | {venue[:40]} | {doi}",
                  flush=True)
            out.append(dict(query=q, year=year, type=typ, title=title,
                            venue=venue, doi=doi))
        time.sleep(0.4)
    with open("/home/user1/projects/code/experiments/ref_candidates.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n[saved] ref_candidates.json", flush=True)


if __name__ == "__main__":
    main()
