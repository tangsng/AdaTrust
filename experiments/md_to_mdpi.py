#!/usr/bin/env python3
"""md_to_mdpi.py — convert paper.md (AdaTrust) to MDPI Information manuscript.tex."""
import re
import os

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "paper.md")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "mdpi",
                   "manuscript.tex")

t = open(SRC, encoding="utf-8").read()

# ---------------------------------------------------------------- unicode map
UNI = {
    "—": "---", "–": "--", "−": "$-$", "×": "$\\times$", "→": "$\\rightarrow$",
    "≈": "$\\approx$", "≥": "$\\geq$", "≤": "$\\leq$", "✗": "$\\times$",
    "✓": "$\\checkmark$", "·": "$\\cdot$", "⌈": "$\\lceil$", "⌉": "$\\rceil$",
    "∞": "$\\infty$", "≜": "$\\triangleq$", "∈": "$\\in$", "∉": "$\\notin$",
    "≠": "$\\neq$", "∝": "$\\propto$", "⟨": "$\\langle$", "⟩": "$\\rangle$",
    "θ": "$\\theta$", "λ": "$\\lambda$", "Δ": "$\\Delta$", "κ": "$\\kappa$",
    "ρ": "$\\rho$", "γ": "$\\gamma$", "ω": "$\\omega$", "δ": "$\\delta$",
    "ϵ": "$\\epsilon$", "ε": "$\\varepsilon$", "φ": "$\\phi$", "π": "$\\pi$",
    "σ": "$\\sigma$", "τ": "$\\tau$", "ξ": "$\\xi$", "η": "$\\eta$",
    "μ": "$\\mu$", "Σ": "$\\Sigma$", "Π": "$\\Pi$", "ℓ": "$\\ell$",
    "%": "\\%", "±": "$\\pm$", "…": "\\ldots", "’": "'",
    "é": "\\'e", "è": "\\`e", "ê": "\\^e", "ć": "\\'c", "ž": "\\v{z}",
    "⁵": "$^5$", "⁷": "$^{-7}$", "⁻": "$^{-}$", "¹": "$^1$",
}

# ASCII fallbacks for code blocks (verbatim cannot carry LaTeX math)
CODE_ASCII = {
    "←": "<-", "α": "alpha", "θ": "theta", "β": "beta", "−": "-",
    "φ": "phi", "̂": "^", "λ": "lambda", "ε": "epsilon", "Σ": "sum",
    "∈": "in", "·": "*", "π": "pi", "…": "...", "′": "'", "Â": "A^",
    "η": "eta", "κ": "kappa", "γ": "gamma", "δ": "delta", "∅": "empty",
    "∞": "inf", "≥": ">=", "τ": "tau", "∪": "union", "²": "^2",
}

# full figure captions (from _build_figs.py; LaTeX adds "Figure N." itself)
FIG_CAPTIONS = {
    1: ("AdaTrust dual-loop architecture: the fast trust loop (BayesElect) "
        "governs committee membership per block; the slow DRL loop "
        "(AdaptSwitch) governs protocol identity and configuration per epoch."),
    2: ("Epoch flow with the bounded-overlap handover invariant: observe, "
        "decide, elect, drain-and-handover, execute."),
    3: ("RQ1 headline throughput on the live chains: per-scheme mean TPS and "
        "per-round timelines (6 paired rounds, interleaved on chain A)."),
    4: ("RQ2: (a) committee-reconfiguration cost vs.\\ batch size (3 reps, "
        "range bars); (b) throughput vs.\\ committee size k (3 serial "
        "repetitions, mean $\\pm$ SD with per-rep dots)."),
    5: ("RQ3 silence-attack timelines (attack at t$=$300 s): trust-score "
        "decay, detection, and committee healing (k 4$\\rightarrow$13) for "
        "AdaTrust vs.\\ StaticTM."),
    6: ("RQ4 ablation: (a) real-chain variants (n$=$3 runs); (b) "
        "simulator-side terminal-window throughput of the trained variants "
        "(1 seed), including the offline w/o-switch-cost collapse to a "
        "static corner."),
}

# roman -> arabic for in-text table references (LaTeX numbers tables 1,2,...)
ROMAN2ARABIC = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
                "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10"}


def unicode_fix(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


# ---------------------------------------------------------------- front matter
title = re.search(r'^# (.+)$', t, re.M).group(1).strip()
abs_m = re.search(r'\*\*Abstract\*\*—(.+?)\*\*Index Terms\*\*—', t, re.S)
abstract = abs_m.group(1).strip().replace("\n", " ")
abstract = abstract.replace("&", "\\&").replace("#", "\\#")
abstract = unicode_fix(abstract)
kw_m = re.search(r'\*\*Index Terms\*\*—(.+?)\n\n---', t, re.S)
keywords = kw_m.group(1).strip()

body_start = t.index("\n---\n", t.index("**Index Terms**")) + len("\n---\n")
body = t[body_start:]
ref_idx = body.index("## References")
main_body = body[:ref_idx]
refs_body = body[ref_idx + len("## References"):]

# Remove in-body Acknowledgments / Data Availability sections (MDPI uses
# back-matter macros \funding / \dataavailability instead).
main_body = re.sub(r'\n## Acknowledgments\n\n.*?(?=\n## (Appendix|Data))', '\n', main_body, flags=re.S)
main_body = re.sub(r'\n## Data Availability Statement\n\n.*?(?=\n## Appendix|\Z)', '\n', main_body, flags=re.S)


# ---------------------------------------------------------------- MDPI structure
def restructure(s):
    """Map paper.md sections to MDPI order:
    Introduction, Related Work, Materials and Methods (3.1-3.4), Results,
    Discussion, Conclusions."""
    out, in_methods = [], False
    for line in s.split('\n'):
        m = re.match(r'^## (\d+)\.?\s*(.+)$', line)
        if m:
            num, title = int(m.group(1)), m.group(2)
            if num == 3:
                out += ['## Materials and Methods', '', '### ' + title]
                in_methods = True
            elif num in (4, 5, 6):
                out += ['### ' + title]
            elif num == 7:
                out += ['## Results']
                in_methods = False
            elif num == 8:
                out += ['## Discussion']
            elif num == 9:
                out += ['## Conclusions']
            else:
                out.append(line)
            continue
        m2 = re.match(r'^### (\d+)\.\d+\.?\s*(.+)$', line)
        if m2:
            if in_methods:
                out.append('#### ' + m2.group(2))  # Methods subsections -> subsubsection
            else:
                out.append(line)
            continue
        out.append(line)
    return '\n'.join(out)


def remap_sections(s):
    """Rewrite Section N / Section N.M / Sections N--M cross-references."""
    top = {'3': '3.1', '4': '3.2', '5': '3.3', '6': '3.4', '7': '4', '8': '5', '9': '6'}

    def map_token(tok):
        tok = tok.strip()
        if '.' in tok:
            n, m = tok.split('.', 1)
            return (top[n] + '.' + m) if n in top else tok
        return top.get(tok, tok)

    # range references: Sections N.M--N.K  (en-dash already converted to --)
    s = re.sub(r'(Sections?)\s+(\d+(?:\.\d+)?)\s*--\s*(\d+(?:\.\d+)?)',
               lambda mm: '%s %s--%s' % (mm.group(1), map_token(mm.group(2)), map_token(mm.group(3))),
               s)
    # single: Section N.M then Section N
    s = re.sub(r'Section (\d+)\.(\d+)',
               lambda mm: 'Section ' + map_token('%s.%s' % (mm.group(1), mm.group(2))), s)
    s = re.sub(r'Section (\d+)\b(?!\.)',
               lambda mm: 'Section ' + map_token(mm.group(1)), s)
    return s


main_body = restructure(main_body)

ref_entries = re.findall(r'^\[(\d+)\]\s+(.*?)(?=\n\n\[|\Z)', refs_body, re.S | re.M)


def ref_fix(txt):
    txt = txt.replace("–", "--").replace("—", "---").replace("−", "--")
    txt = re.sub(r'\*([^*]+)\*', r'\\emph{\1}', txt)
    txt = txt.replace("&", "\\&").replace("_", "\\_").replace("#", "\\#")
    out, open_q = [], True
    for ch in txt:
        if ch == '"':
            out.append('``' if open_q else "''")
            open_q = not open_q
        else:
            out.append(ch)
    return unicode_fix(''.join(out))


ref_texts = sorted([(int(n), ref_fix(txt.strip().replace("\n", " "))) for n, txt in ref_entries])

# ---------------------------------------------------------------- body transform
table_captions = []
table_idx = [0]


def capture_table_title(mm):
    table_captions.append(mm.group(2))  # caption text only; LaTeX adds "Table N."
    return ""


def table_repl(mm):
    header = mm.group(1).strip().strip('|')
    rows = mm.group(2).strip().split('\n')
    ncol = header.count('|') + 1
    # first column sized to its content (l); remaining columns share the width (X)
    colspec = '@{}l' + 'X' * (ncol - 1) + '@{}'
    def cells(line):
        # content-level & already escaped globally; split on | only
        return [c.strip() for c in line.strip().strip('|').split('|')]
    hdr = ' & '.join(unicode_fix(c) for c in cells(header))
    body_lines = [' & '.join(unicode_fix(c) for c in cells(r)) + r' \\' for r in rows]
    cap = table_captions[table_idx[0]] if table_idx[0] < len(table_captions) else "Table"
    table_idx[0] += 1
    # all tables extend into the left margin for symmetric page margins
    return ('\\begin{table}[H]\n\\centering\n\\caption{' + cap + '}\n'
            '\\small\n'
            '\\begin{adjustwidth}{-\\extralength}{0cm}\n'
            '\\begin{tabularx}{\\linewidth}{' + colspec + '}\n'
            '\\toprule\n' + hdr + r' \\' + '\n\\midrule\n'
            + '\n'.join(body_lines) + '\n\\bottomrule\n'
            '\\end{tabularx}\n\\end{adjustwidth}\n\\end{table}')


def math_repl(mm):
    inner = re.sub(r'\\tag\{\d+\}', '', mm.group(1)).strip()
    if '\\\\' in inner:
        return '\\begin{multline}\n' + inner + '\n\\end{multline}'
    return '\\begin{equation}\n' + inner + '\n\\end{equation}'


def fig_repl(mm):
    num = int(mm.group(1))
    path = mm.group(3)
    base = os.path.splitext(os.path.basename(path))[0]
    cap = FIG_CAPTIONS.get(num, mm.group(2))
    return ('\\begin{figure}[H]\n\\centering\n'
            '\\begin{adjustwidth}{-\\extralength}{0cm}\n'
            '\\centering\n'
            '\\includegraphics[width=\\linewidth]{figures/%s.pdf}\n'
            '\\caption{%s}\n'
            '\\end{adjustwidth}\n\\end{figure}' % (base, cap))


def body_transform(s):
    # 0. remove markdown horizontal rules (---)
    s = re.sub(r'^\s*---+\s*$', '', s, flags=re.M)
    # 0.3 algorithm blocks: "**Algorithm N — Title**" + ```code``` -> MDPI Algorithm env
    def algorithm_repl(mm):
        title = unicode_fix(mm.group(1))
        content = mm.group(2)
        for u, a in CODE_ASCII.items():
            content = content.replace(u, a)
        return ('\\begin{adjustwidth}{-\\extralength}{0cm}\n'
                '\\begin{Algorithm}[%s]\\ \\\\\n'
                '\\normalfont\n'
                '\\begin{verbatim}\n%s\\end{verbatim}\n'
                '\\end{Algorithm}\n\\end{adjustwidth}' % (title, content))
    s = re.sub(r'\*\*Algorithm \d+ — (.+?)\*\*\s*\n```\n(.*?)```',
               algorithm_repl, s, flags=re.S)
    # 0.5 code blocks -> verbatim (ASCII math, verbatim cannot carry LaTeX);
    #    extend into the left margin for symmetric page margins
    def code_block_repl(mm):
        content = mm.group(1)
        for u, a in CODE_ASCII.items():
            content = content.replace(u, a)
        return ('\\begin{adjustwidth}{-\\extralength}{0cm}\n'
                '\\begin{verbatim}\n' + content + '\\end{verbatim}\n'
                '\\end{adjustwidth}')
    s = re.sub(r'```\n(.*?)```', code_block_repl, s, flags=re.S)
    # 1. display math -> equation (BEFORE inline-math protection)
    s = re.sub(r'\$\$\s*(.+?)\s*\$\$', math_repl, s, flags=re.S)
    # 2. protect inline math so *, &, _ etc. inside $...$ are untouched
    math_spans = []
    def _prot(mm):
        math_spans.append(mm.group(0))
        return '\x00M%d\x00' % (len(math_spans) - 1)
    s = re.sub(r'\$[^$\n]+\$', _prot, s)
    # 3. escape content-level & (before tables; column separators are
    # regenerated by table_repl afterwards)
    s = s.replace('&', '\\&')
    # 4. table captions (BEFORE bold)
    s = re.sub(r'\*\*TABLE ([IVX]+) — (.+?)\*\*', capture_table_title, s)
    # 5. figures (strip surrounding markdown * from `*(Fig. N: ...)*`)
    s = re.sub(r'\*\(Fig\. (\d+): (.+?) — `([^`]+)`\)\*', fig_repl, s)
    # 6. pipe tables
    s = re.sub(r'\|(.+)\|\n\|[-: |]+\|\n((?:\|.+\|\n)+)', table_repl, s)
    # 5. section headers (numbered or plain)
    s = re.sub(r'^## (?:\d+\.?\s*)?(.+)$', r'\\section{\1}', s, flags=re.M)
    s = re.sub(r'^### (?:\d+\.\d+\.?\s*)?(.+)$', r'\\subsection{\1}', s, flags=re.M)
    s = re.sub(r'^#### (.+)$', r'\\subsubsection{\1}', s, flags=re.M)
    # 6. citations ([1-9] excludes math intervals like [0,1])
    s = re.sub(r'\[([1-9]\d*(?:,\s*[1-9]\d*)*)\]',
               lambda mm: '\\cite{' + ','.join('ref' + x.strip() for x in mm.group(1).split(',')) + '}',
               s)
    # 6.5 table refs: roman -> arabic (LaTeX numbers tables 1,2,...)
    s = re.sub(r'Table ([IVX]+)\((a|b)\)',
               lambda mm: 'Table ' + ROMAN2ARABIC.get(mm.group(1), mm.group(1)) + '(' + mm.group(2) + ')', s)
    s = re.sub(r'Table ([IVX]+)\b',
               lambda mm: 'Table ' + ROMAN2ARABIC.get(mm.group(1), mm.group(1)), s)
    # 7. bold / italic
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<![*\\$])\*([^*\n]+?)\*(?![\*$])', r'\\emph{\1}', s)
    # 8. backticks -> texttt (escape _ inside filenames)
    def tt_repl(mm):
        content = mm.group(1).replace('_', '\\_')
        return '\\texttt{' + content + '}'
    s = re.sub(r'`([^`]+)`', tt_repl, s)
    # 9. unicode
    s = unicode_fix(s)
    # 10. restore protected inline math
    s = re.sub(r'\x00M(\d+)\x00', lambda mm: math_spans[int(mm.group(1))], s)
    # 10.5 remove manual proof-end black squares (not MDPI style)
    s = s.replace('$\\blacksquare$', '')
    return s


main_body_t = body_transform(main_body)
main_body_t = remap_sections(main_body_t)

# ---------------------------------------------------------------- assemble
front = r'''%%=============================================================%
%%  AdaTrust — MDPI Electronics
%%=============================================================%

\documentclass[electronics,article,submit,moreauthors]{Definitions/mdpi}

\firstpage{1}
\makeatletter \setcounter{page}{\@firstpage} \makeatother
\pubvolume{1}
\issuenum{1}
\articlenumber{0}
\pubyear{2026}
\copyrightyear{2026}
\datereceived{ }
\daterevised{ } % Comment out if no revised date
\dateaccepted{ }
\datepublished{ }

'''

front += "\\Title{" + title + "}\n\n"
front += r'''\Author{Song Tang $^{1,2,3}$\orcidA{}, Zhigang Jin $^{1,*}$\orcidB{} and Zhiqiang Wang $^{2,3}$\orcidC{}}

\AuthorNames{Song Tang, Zhigang Jin and Zhiqiang Wang}

\address{%
$^{1}$ \quad School of Electrical and Information Engineering, Tianjin University, Tianjin 300072, China\\
$^{2}$ \quad Institute of Applied Mathematics, Hebei Academy of Sciences, Shijiazhuang 050081, China\\
$^{3}$ \quad Information Security Authentication Technology Innovation Center of Hebei Province, Shijiazhuang 050081, China}

\corres{Correspondence: zgjin@tju.edu.cn (Z.J.)}

\newcommand{\orcidauthorA}{0000-0001-9048-0738} % Song Tang
\newcommand{\orcidauthorB}{0000-0001-5777-569X} % Zhigang Jin
\newcommand{\orcidauthorC}{0009-0005-5718-6867} % Zhiqiang Wang

'''

front += "\\abstract{" + abstract + "}\n\n"
front += "\\keyword{" + keywords + "}\n\n"
front += "\\emergencystretch=3em\n"
front += "\\begin{document}\n\n"

back = r'''
\vspace{6pt}

\authorcontributions{Conceptualization, S.T. and Z.J.; methodology, S.T.; software, S.T.; validation, S.T. and Z.W.; formal analysis, S.T.; investigation, S.T. and Z.W.; resources, Z.J.; data curation, S.T.; writing---original draft preparation, S.T.; writing---review and editing, S.T., Z.J. and Z.W.; visualization, S.T.; supervision, Z.J.; project administration, Z.J.; funding acquisition, Z.J. All authors have read and agreed to the published version of the manuscript.}

\funding{This research was funded by the Hebei Provincial Science and Technology Program Project (Grant No. 25360301D): Research and Application Demonstration of Key Technologies for Public Data Authorization and Operation Based on Trusted Data Space.}

\institutionalreview{Not applicable.}

\informedconsent{Not applicable.}

\dataavailability{The source code and raw data that support the findings of this study are openly available in AdaTrust at \url{https://github.com/tangsng/AdaTrust}.}

\conflictsofinterest{The authors declare no conflicts of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.}

\useofartificialintelligence{AI or AI-assisted tools were not used in drafting any aspect of this manuscript.}

\abbreviations{Abbreviations}{%
The following abbreviations are used in this manuscript:\\
\noindent
\begin{tabular}{@{}ll}
BFT & Byzantine fault tolerance\\
DRL & deep reinforcement learning\\
PPO & proximal policy optimization\\
TPS & transactions per second\\
MDP & Markov decision process
\end{tabular}
}

\isAPAandChicago{}{%
\reftitle{References}
\begin{adjustwidth}{-\extralength}{0cm}
\begin{thebibliography}{99}
'''
for num, txt in ref_texts:
    back += f"\\bibitem[{{{num}}}]{{ref{num}}} {txt}\n"
back += r'''
\end{thebibliography}
\end{adjustwidth}
}

\end{document}
'''

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(front)
    f.write(main_body_t)
    f.write("\n")
    f.write(back)

print("manuscript.tex written:", len(front) + len(main_body_t) + len(back), "chars")
print("references:", len(ref_texts), "| tables:", main_body_t.count(r'\begin{table}'),
      "| figures:", main_body_t.count(r'\begin{figure}'),
      "| equations:", main_body_t.count(r'\begin{equation}'))
print("table captions:", len(table_captions))
