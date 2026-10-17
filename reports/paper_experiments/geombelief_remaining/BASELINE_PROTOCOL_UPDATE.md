# GeomBelief baseline protocol update

Official source snapshots, commit SHAs and fetch failures are in baseline_source_manifest.json. Original results and baselines were not retrained. Existing source evidence remains in ../baseline_protocol_audit.md and ../dataset_version_audit.json.

|Method|Evidence category|Remaining comparability issue|
|---|---|---|
|MGP|Official code and recovered weight; local variants separately indexed|Goal/progress postprocessing and cache protocols differ; no relabeling as strict reproduction|
|HETT|Official repository and released checkpoint evidence|Local research checkout contains modifications; best_val_unseen weight-selection provenance remains explicit|
|HTNav|Official source; local independent training results|Official README still says code upload in progress; no verified official trained weight was found in that README. Local model is not an official-weight reproduction|
|GeoNav (DeepSeek Flash)|Model-substituted API experiment only|Not GPT-4o, single local run versus paper multiple runs; original Test 5311 must not be merged with revised Test5281|
|FlightGPT|Official code and public model reference|Full local official-weight evaluation not complete; no number invented|

Three copied CSVs keep paper-reported, local common-evaluator and other/API protocols separate. They are evidence views, not newly reproduced results.

All local GeomBelief metrics use XY final distance ≤20m; OSR uses any actual visited XY; SPL = success × straight-line start-to-goal distance / max(actual XY path length, straight-line start-to-goal distance). GT enters scoring only. Stop probability, predicted-goal arrival gate and evaluator success are different concepts.

HTNav local htnav/evaluate.py uses XY distances but its SPL optimal length is the generated shortest teacher trajectory, not necessarily Euclidean start-to-goal distance. GeoNav vlnce/evaluate.py uses a 3D final-distance branch and teacher path length, while scenegraphnav/evaluate.py uses XY; the module actually used must be recorded rather than assigning a repository-wide metric. Native '<20' versus common '<=20' is also retained in audit. FlightGPT includes adapted evaluator modules; do not infer formula from metric names.

Current revised episode counts are 2470/2697/5281, versus original2498/2826/5311. Prior file-level audit found311 total removed episodes; this is not interchangeable with every paper's stated revision count. No official dataset-byte identity is asserted without matching source SHA. Inputs differ across methods: language/name+map geometry, observed RGB-D/history, global map images and API model prior knowledge are reported separately.

Official links: [MGP/CityNav](https://github.com/water-cookie/citynav), [HETT](https://github.com/crotonyl/HETT), [HTNav](https://github.com/tyrofanfan/HTNav), [GeoNav](https://github.com/Xhtshr/geonav-official), [FlightGPT](https://github.com/Uavln/FlightGPT). No excluded historical research method was added to the comparison.
