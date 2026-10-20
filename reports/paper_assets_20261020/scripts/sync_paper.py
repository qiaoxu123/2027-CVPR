"""Copy lightweight derived figures/evidence into the requested manuscript repo."""
from common import *
import shutil,subprocess

def main():
    target=PAPER/'figures/citynav';target.mkdir(parents=True,exist_ok=True)
    for file in (OUT/'figures').glob('*'):
        if file.is_file():shutil.copyfile(file,target/file.name)
    source=target/'source';(source/'scripts').mkdir(parents=True,exist_ok=True);(source/'fields').mkdir(exist_ok=True)
    for name in ('cases.jsonl','geometry_case.json','public_map_context.json','observation_index.json','replay_manifest.json'):
        shutil.copyfile(OUT/name,source/name)
    for file in (OUT/'fields').glob('*'):
        if file.suffix in ('.json','.npz'):shutil.copyfile(file,source/'fields'/file.name)
    for name in ('common.py','plot_figures.py'):shutil.copyfile(OUT/'scripts'/name,source/'scripts'/name)
    backup=PAPER/'figures/overview_conceptual.tex'
    if not backup.exists():shutil.copyfile(PAPER/'figures/overview.tex',backup)
    backup=PAPER/'figures/visual_evidence_conceptual.tex'
    if not backup.exists():shutil.copyfile(PAPER/'figures/visual_evidence.tex',backup)
    (PAPER/'figures/overview.tex').write_text(r'''% Real frozen-model evidence; conceptual predecessor kept separately.
\begin{minipage}{\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/citynav/figure1_overview.pdf}
\vspace{-0.7em}
\captionof{figure}{\textbf{GeomBelief on a real CityNav episode.} The archived C-epoch1 model uses language and named map geometry to form $B_0$, updates its belief from actual observations, and executes bounded goal-directed navigation. Episode \texttt{birmingham\_block\_5:8:4}; observed footprints at decisions 0 and 7; final NE 11.95\,m. The visual correction in this example is small (belief L1 $=0.0262$); success alone does not establish a visual gain. Public artwork shows measured footprints; real RGB crops are indexed for local reproduction pending underlying image redistribution terms. GT is used only for evaluation/plotting.}
\label{fig:overview}
\end{minipage}
''')
    (PAPER/'figures/visual_evidence.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/citynav/figure3_visual_updates.pdf}
\caption{\textbf{Measured visual belief updates with the separately calibrated residual head.} Cases A/B/C respectively correct a wrong Top-1 (22.22 to 11.95\,m), reinforce a correct hypothesis (GT-region probability 0.298 to 0.476), and degrade localization (14.86 to 32.05\,m). Differences are computed from normalized probabilities; per-step L1, KL and Top-1 changes are released. Zero, shuffled and disabled-residual controls replay identical real-trajectory observation prefixes; they are not their independent navigation rollouts. Shuffle can retain the same Top-1 in an individual example, limiting semantic attribution. Outcome-selected validation illustrations, not aggregate effect estimates. Map attribution: OpenStreetMap contributors / CityNav.}
\label{fig:evidence}
\end{figure*}
''')
    (target/'geometry.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/citynav/figure2_geometry.pdf}
\caption{\textbf{Real language--map grounding example.} Name/type-only, shuffled and correct geometry give Top-1 errors of 358.29, 22.29 and 12.02\,m for query \texttt{birmingham\_block\_5:8:5:0}. The three fixed-seed heads were separately trained with matched budgets; this is not a frozen-main-weight intervention. Solid polygons are recipient map context; dotted cyan polygons are the shuffled donor input. Numeric geometric measurements use source world coordinates and do not assume that $+Y$ is geographic north. Outcome-selected illustration. Source/weight SHAs accompany the figure.}
\label{fig:real_geometry}
\end{figure*}
''')
    (target/'navigation.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/citynav/figure4_navigation.pdf}
\caption{\textbf{Actual navigation successes and failures of the archived main checkpoint.} Two successes and four failures are selected in source order within predefined outcome groups. Blue paths are executed UAV trajectories, stars denote GT, yellow crosses the final selected goal, and red squares the actual final position. Visited-but-failed means OSR$=1$ and SR$=0$; it does not alone identify a learned-Stop causal failure. Exact episode identities, frame poses, checkpoint SHA and predictions are indexed in the evidence bundle. Map attribution: OpenStreetMap contributors / CityNav.}
\label{fig:real_navigation}
\end{figure*}
''')
    method=PAPER/'sec/4_method.tex';text=method.read_text();text=text.replace('\\input{figures/visual_evidence}','% Measured visual evidence appears with the corresponding experiment.');method.write_text(text)
    exp=PAPER/'sec/5_experiments.tex';text=exp.read_text()
    if '\\input{figures/citynav/geometry}' not in text+(PAPER/'sec/3_problem.tex').read_text():
        text=text.replace('\\subsection{Independent Contribution of Map Geometry}','\\input{figures/citynav/geometry}\n\n\\subsection{Independent Contribution of Map Geometry}')
    if '\\input{figures/visual_evidence}' not in text:
        text=text.replace('\\subsection{Observation-Conditioned Visual Correction}','\\input{figures/visual_evidence}\n\n\\subsection{Observation-Conditioned Visual Correction}')
    if '\\input{figures/citynav/navigation}' not in text:
        text=text.replace('\\subsection{Goal Selection, Waypoints, and Stopping}','\\input{figures/citynav/navigation}\n\n\\subsection{Goal Selection, Waypoints, and Stopping}')
    if '\\paragraph{Traceable qualitative evidence.}' not in text:
        text=text.replace('\\subsection{Limitations}',r'''\paragraph{Traceable qualitative evidence.}
Figures~\ref{fig:real_geometry}, \ref{fig:evidence}, and \ref{fig:real_navigation} use measured validation records. Frozen replay matches archived Top-16 goals, selected targets, and Stop probabilities for nine episode--model cases (eight distinct episodes; 80 observed-frame instances). The released audit contains 310 per-step/comparison rows of belief L1, KL divergence, and Top-1 displacement. Offline XY rescoring of all 10,448 archived evaluation episodes reproduces the formal NE/SR/OSR/SPL; no new Test navigation or checkpoint selection was performed. In the saved weak-projection visibility partition, 332/2,470 Seen episodes are strictly visible at the start, with SR 51.51\%; this is a descriptive subset, not a human-verified grounding score. RGB crops remain locally indexed until underlying raster redistribution terms are established; the submitted derived figures show actual observed footprints.

\subsection{Limitations}''')
    exp.write_text(text)
    (target/'reproduce.py').write_text('''"""Rebuild all four no-RGB figures from the bundled real source arrays."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(root/'source/scripts/plot_figures.py'),'--output-dir',str(root),*sys.argv[1:]],check=True)
''')
    (target/'README.md').write_text('''# Real CityNav evidence

PDF is the vector publication artifact; PNG is the preview. JSON sidecars trace each figure to episode, data version, source predictions, checkpoints and generation command. `source/` contains the lightweight fields, traces for nine episode--model cases (eight distinct validation episodes), map context, frame indices and plotting scripts. No raw RGB, large dataset or checkpoint is included.

From the manuscript repository, rebuild using Python with numpy, matplotlib and shapely:

```bash
python figures/citynav/reproduce.py
```

Standalone figure1 and figure4 use the formal frozen C checkpoint; figure2 uses three separately matched trained heads; figure3 uses the independently calibrated visual residual. They must not be combined into one new model or used to claim an unbiased visual gain. The extra preserved architecture illustration has its own live manuscript counter; use LaTeX labels and JSON sidecars rather than assuming standalone filenames equal every paper figure number. All scenes and traces are real data; old conceptual city/result drawings remain tracked as history and are not live empirical figures.

Actual RGB-containing figures stay on the lab server under the experiment bundle's `local_only/figures/`. See `notes/visual_evidence_manifest.md` for image terms and full reproduction instructions. Map attribution: CityNav / OpenStreetMap contributors, https://www.openstreetmap.org/copyright.
''')
    shutil.copyfile(OUT/'BASELINE_AUDIT.md',PAPER/'notes/baseline_verification.md')
    shutil.copyfile(OUT/'BIBLIOGRAPHY_AUDIT.md',PAPER/'notes/bibliography_verification.md')
    shutil.copyfile(OUT/'baseline_results.csv',PAPER/'notes/baseline_results.csv')
    shutil.copyfile(OUT/'bibliography_audit.csv',PAPER/'notes/bibliography_audit.csv')
    print('Synced lightweight evidence and four real figures to',PAPER)

if __name__=='__main__':main()
