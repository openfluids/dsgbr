# README Banner v1 Prompt

Generated asset: `assets/readme-banner-v1.png`

Tool/model: Built-in `image_gen` tool

Repo analysis:
- DSGBR is a scientific Python package for spectral peak detection in frequency-domain signals.
- The README and docs emphasize noisy PSDs, a SEARCH/BASELINE ratio, rolling-median baseline separation, and accepted peaks.
- The proof surface includes deterministic synthetic figures, benchmark comparisons against `scipy.signal.find_peaks`, and parameter-sensitivity documentation.
- The visual identity should feel like restrained scientific software branding, not generic Python packaging.

Repository homogeneity exceptions:
- Open Dependabot PR #9, `chore(deps): bump the actions group across 1 directory with 7 updates`, left open for this one-off banner run.
- Open Dependabot PR #4, `chore(deps-dev): bump the pip-dependencies group with 2 updates`, left open for this one-off banner run.

Prompt:

```text
Create a polished wide GitHub README banner for an open-source repository named "DSGBR".

Use case: stylized-concept
Asset type: GitHub README top banner, wide landscape, approximately 3:1 aspect ratio
Primary request: A sophisticated visual identity for DSGBR, a scientific Python package for spectral peak detection in noisy frequency-domain signals. The banner must communicate robust peak detection, baseline separation, and reproducible scientific computing.
Scene/backdrop: A refined technical visualization inspired by a log-frequency power spectral density plot: a dense noisy spectrum trace, a smoother baseline curve below it, and several clean highlighted accepted peaks rising above the background. Use abstract plot geometry and scientific data-visualization motifs, not a literal screenshot.
Subject: Noisy PSD signal, rolling-median baseline, short-scale search trace, accepted spectral peaks, and subtle benchmark/validation structure without readable numeric values.
Style: refined technical branding, clean scientific/software aesthetic, high contrast, professional, not cartoonish.
Composition: include the exact project name "DSGBR" as large prominent readable text inside the banner, centered or slightly left of center, with generous spacing. Arrange the spectrum traces and highlighted peaks around the text without covering it.
Evidence motifs: include subtle references to synthetic benchmark scenarios, support-series overlays, and peak markers, but no readable fake numbers, tables, badges, or UI controls.
Text: render exactly "DSGBR" and no other readable text.
Constraints: no third-party logos, no Python logo, no fake badges, no fabricated metrics, no watermarks, no mascots, no extra words, no misspellings.
Output intent: professional README banner suitable for GitHub, saved as assets/readme-banner-v1.png.
```

Post-processing:
- Copied the selected generated PNG from `/home/rfrantz/.codex/generated_images/019eb58e-d3b0-7ba0-98e2-db3c21c31cdb/ig_071eae3d56ffacaf016a2a76b3c7a481919f6a6a9e8ed969b6.png` to `assets/readme-banner-v1.png`.
- No local overlay, crop, resize, or color correction.
