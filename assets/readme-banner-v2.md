# README Banner v2

Asset: `assets/readme-banner-v2.jpg` (1408x469, 3:1, 184 KB)

Tool/model: xAI Grok CLI, built-in `image_gen` tool, plus local compositing.

Replaces `readme-banner-v1.png`. This version adopts the shared language now
used across the openfluids repositories — `dynachaos`, `fftkit`, `chaos-atlas`,
`openmodalpy` — all 3:1, all on a charcoal ground with a warm off-white
lowercase wordmark and cyan/teal structure with coral accents.

## Approach

The wordmark is **not** generated. Image models render short lowercase words
unpredictably, and accepting whatever letterforms come back is most of what
makes a generated banner look cheap. The artwork is generated deliberately
textless and the type is set locally in Lato Light, sized to a fixed fraction of
the frame width so names of different lengths carry comparable optical weight.

## Subject

A dense, noisy spectral thicket in teal, decaying along a sloped envelope; a
smooth continuous baseline curve threading through it at the level of the noise;
and a set of sharp coral peaks standing clearly above that baseline, picked out
from their neighbours.

That is the algorithm, not a decoration around it. DSGBR detects peaks by ratio
against a rolling baseline rather than against a fixed prominence threshold,
which is what lets it work on a spectrum that slopes over several decades — the
case where a fixed threshold either drowns in low-frequency power or misses
everything at high frequency.

## Prompt (artwork only, no text)

```text
A stunning abstract scientific artwork, wide 2:1 landscape: a forest of fine
vertical luminous spectral lines of varying height rising from a baseline, dense
and crowded, their overall envelope decaying smoothly from left to right along a
power-law curve. Most lines are dim teal and blend into a noisy thicket, but a
regularly spaced subset stands taller, sharper and brilliantly lit in coral and
amber, clearly picked out from their neighbours, as though selected. A smooth
continuous curve sweeps through the thicket at the level of the noise,
separating the selected peaks above from the clutter below. Deep near-black
charcoal ground, volumetric glow, atmospheric depth of field, fine film grain,
rich deep blacks and luminous highlights. Cinematic, precise, elegant,
expensive, gallery-quality scientific data art. ABSOLUTELY NO TEXT [...] Leave
the left third dark, calm and completely empty as negative space.
```

## Post-processing

- Returned 1408x704, centre-cropped to 1408x469 for the family's 3:1 aspect.
  `image_gen` rejects a 3:1 request and falls back to 2:1, so prompts must state
  that the image will be letterboxed and keep all subject matter in the central
  band.
- Wordmark composited locally: Lato Light, auto-sized to 20% of frame width
  (107 px here — `dsgbr` is short), tracking 6% of point size, warm off-white
  `#F7F3EC`, with a wide blurred dark halo underneath.
- JPEG q95, no chroma subsampling: 184 KB, against 2.0 MB for the v1 PNG.

## Rejected alternative

- **Peak comb with markers** — an explicit stem plot of detected peaks, each
  crowned with a glowing marker above a baseline curve. The most literal reading
  of the algorithm, and cleaner, but the stems ran into the left third and
  collided with the wordmark, and the scatter read thin next to the selected
  version.
