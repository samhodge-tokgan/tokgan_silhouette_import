# tokgan_silhouette_import

Import Tokgan ML Splines into Silhouette from Boris FX.

## Usage

1. Create a Roto Session from the frames folder.
2. Select the Roto Node.
3. Run the script (`data/tokgan_json_to_fxs.py`).
4. Browse to the JSON file with the Qt file dialog.

## Schema compatibility

Accepts `lozenge_bezier_anim` schema **v2 and v3**. The v3 additions
(top-level `camera` + `persons` reference-frame blocks, from
Rotobot-Next issue #279) are read-and-ignored by this converter — the
Silhouette FXS output format doesn't yet consume the plate-camera or
person-root data, so new fields pass through harmlessly. Older v2
JSONs keep working unchanged.

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
