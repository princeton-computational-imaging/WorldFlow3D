# Sample Waymo maps

Three illustration-only Waymo Open Dataset scenarios, provided so the Waymo
inference example runs out of the box. Each file is a ScenarioMax-converted
scenario **trimmed** to only what the model conditions on — the static road
network (lanes, road lines/edges, crosswalks, speed bumps) plus parked vehicles.
All moving-agent trajectories and traffic-light states have been removed.

| File (`--segment` id) | Scene |
|-----------------------|-------|
| `WOD_v1.4.3_1020629352036937500.json` (`1020629352036937500`) | urban intersection |
| `WOD_v1.4.3_1027514466074967382.json` (`1027514466074967382`) | multi-lane road |
| `WOD_v1.4.3_1172406780360799916.json` (`1172406780360799916`) | residential street |

Run (no `--data-root` needed — the WOD id matches the file directly):

```bash
python -m worldflow3d.scripts.generate_waymo \
    --repo pci-lab/worldflow3d-waymo --stage waymo-coarse --refine-stage waymo-refine \
    --segment 1020629352036937500 --map-dir examples/sample_maps \
    --smaller-map --fraction 0.12 --use-uniform-chunking \
    --output outputs/segment-1020629352036937500.ply
```

## License

These are **De Minimis** extracts of the Waymo Open Dataset, shared under the
Waymo Dataset License Agreement for **Non-Commercial Use**. Downstream use is
subject to the WOD terms. See [`../../WAYMO_NOTICE.md`](../../WAYMO_NOTICE.md)
and <https://waymo.com/open/terms>.

> This data was made using the Waymo Open Dataset, provided by Waymo LLC under
> the Waymo Dataset License Agreement for Non-Commercial Use, available at
> waymo.com/open/terms.
