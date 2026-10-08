# tokgan_silhouette_import

Import Tokgan ML Splines into Silhouette from Boris FX.

Two tools, both copies of the Silhouette integration shipped with Rotobot Next
(see `SYNCED_FROM.txt`):

* **`data/tokgan_json_to_fxs.py`** — command-line converter from a Rotobot
  shape JSON to a native Silhouette `.fxs`. Python 3.9+, standard library
  only. This is the same converter Rotobot Queue uses for its automatic
  `.fxs`.
* **`tokgan_silhouette_import_hierachy.py`** — a Silhouette **Action** that
  imports the JSON into the selected Roto node from inside Silhouette.

## Converter usage

```bash
python data/tokgan_json_to_fxs.py shapes.json [out.fxs] [--frame-offset 0] [--layers] [--no-hierarchy] [--log]
```

Then in Silhouette, import the `.fxs` into a Roto node on the plate.

* `--frame-offset N` — Silhouette frame = JSON frame + N. Rotobot numbers
  frames from 1, so use **`--frame-offset 0`** for footage numbered from 1
  (the default, -1, suits footage numbered from 0).
* `--layers` — group shapes person → region → side.
* `--no-hierarchy` — for a v3 JSON, skip the camera hierarchy below.

Keep `data/_rotobot_hierarchy/` next to the script: it is a vendored copy
of the camera / person decomposition from
[rotobot-nuke](https://github.com/samhodge-tokgan/rotobot-nuke) (MIT).

## Action usage

1. Create a Roto Session from the frames folder.
2. Select the Roto Node.
3. Run the Action (`tokgan_silhouette_import_hierachy.py`).
4. Browse to the JSON file with the Qt file dialog.

## Schema compatibility

Accepts `lozenge_bezier_anim` schema **v2 and v3**. A v2 JSON converts to
shapes exactly as before.

A **v3** JSON (Rotobot Next 0.10.0+, which adds the plate camera and
per-person reference frames) converts to a hierarchy organised for editing:

```
camera                 root layer: its corner pin is the plate camera
                       (the exact perspective solve), keyed every frame
  p0                   layer per person, positioned at the pelvis
    p0_arm_L_forearm   layer per body part: position + rotation from the bone,
                       its shape's points are bone-local
```

* Disable the `camera` layer's transform to see the rig stabilised.
* Move `p0` to move the whole person; adjust a limb on its own layer.
* Missing or failed tracking holds the last good value instead of jumping.
* Before writing, every vertex is recomposed through the layers and checked
  against the source (within 0.05 px); the shapes land exactly where the
  flat conversion puts them.

Checked in Silhouette on a real Rotobot Next 0.10.0 shot. The Action script
does not build this hierarchy yet.

### Reducing keyframe count before import

The converter writes **every** frame of each spline as a Silhouette
FXS keyframe. On a 24 fps action clip that's dense — fine to render,
less fun to hand-tweak.

For a Rotobot-Next v3 JSON, pre-pass through
[`rotobot-undersample`](https://pypi.org/project/rotobot-nuke/) before
running this converter:

```bash
pip install rotobot-nuke        # provides the `rotobot-undersample` CLI
rotobot-undersample shapes.json shapes_reduced.json --preset balanced
python data/tokgan_json_to_fxs.py shapes_reduced.json
```

The `balanced` preset keeps ~ 89% of keyframes on a measured 4-clip
real-plate benchmark (3 pp spread across clips), dropping the ones
that are linear interpolations of their neighbours in a composed
camera + person-root + body-local-articulation metric. See
[`rotobot-nuke/benchmarks/real_results_cross_clip.md`](https://github.com/samhodge-tokgan/rotobot-nuke/blob/main/benchmarks/real_results_cross_clip.md)
for the methodology + fine/coarse alternatives.

The pre-pass is optional — direct conversion still works unchanged.
