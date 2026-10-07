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
