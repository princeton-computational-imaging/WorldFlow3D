# Waymo Open Dataset — License Notice & Attribution

**Prominent notice (required by the Waymo Open Dataset License Agreement):**

> The WorldFlow3D **Waymo models** (the `waymo-coarse`, `waymo-refine`, and
> `waymo-color` checkpoints) and the **sample maps** under [`examples/sample_maps/`](examples/sample_maps/)
> are *Distributed WOD Models* / data extracts developed using the
> [Waymo Open Dataset](https://waymo.com/open). They are released under the
> **Waymo Dataset License Agreement for Non-Commercial Use**, and **any further
> downstream use or modification — including scenes, meshes, or datasets
> generated with the Waymo models — is subject to the terms of that Agreement,
> including its non-commercial restrictions.** A copy of the Agreement is
> available at <https://waymo.com/open/terms>.

This notice applies to recipients with respect to the whole of these Waymo
artifacts. WorldFlow3D's own copyright and the Apache-2.0 terms cover the
**code**, but do **not** override the non-commercial restrictions above.

## Attribution

> This model and these sample maps were made using the Waymo Open Dataset,
> provided by Waymo LLC under the Waymo Dataset License Agreement for
> Non-Commercial Use, available at waymo.com/open/terms.

## What this covers

| Artifact | Where | License |
|----------|-------|---------|
| `waymo-coarse`, `waymo-refine`, `waymo-color` model weights | Hugging Face (Waymo model repo) | Waymo Dataset License Agreement (non-commercial) |
| Sample maps (`WOD_v1.4.3_*.json`) | `examples/sample_maps/` | De Minimis extracts under the WOD terms (see below) |
| Scenes / meshes / datasets generated **with the Waymo models** | (your outputs) | "Distributed WOD Models" — also non-commercial |

The Front3D models are trained on 3D-FRONT (not WOD) and are governed by the
3D-FRONT terms instead — they are **not** covered by this notice. The
`worldflow3d` **code** is Apache-2.0.

## Sample maps — De Minimis basis

The files in `examples/sample_maps/` are small, illustration-only extracts of
the Waymo Open Dataset, provided so the inference examples are runnable. Each is
a single scenario trimmed to only the static road network and parked vehicles
that the model conditions on; all moving-agent trajectories and traffic-light
states have been removed. They are shared under the WOD "De Minimis" clause and
are **not** redistributed through PyPI (they are excluded from the package).

## The Waymo Open Dataset sharing terms (verbatim)

> **Sharing**: Subject to these Terms, you may share as follows:
>
> - **De Minimis**: You may publish small extracts of data taken from the
>   Dataset for purposes of illustration in Your research publications;
> - **Dataset Community Distribution Only**: You may distribute the Dataset and
>   copies and modifications thereof, but only if (i) distribution is limited to
>   those who have registered at waymo.com/open and agreed to these Terms and
>   (ii) You agree not to prepare, initiate, assert, or otherwise support any
>   claim against Waymo or any authorized licensee of the Dataset for violation
>   of any of Your rights protecting the distributed work; and
> - **Model Architectures Non-commercial Distribution Only**: You may publish,
>   in whole or in part, trained model architectures, including weights and
>   biases, developed using the Dataset, and their associated derivatives
>   including new datasets generated therefrom (collectively, "Distributed WOD
>   Models"), provided that You meet the following conditions:
>   - You must include a prominent notice to Your users or recipients stating
>     that (x) Your Distributed WOD Models is released under this Agreement and
>     (y) any further downstream use or modification of such Distributed WOD
>     Models is subject to the Terms of this Agreement, including the
>     non-commercial restrictions of Section 4.
>   - This Agreement shall apply to Your users or recipients with respect to the
>     whole of Your Distributed WOD Models, and You may provide Your own
>     copyright statement and add additional terms and conditions for Your
>     Distributed WOD Models, provided that they do not conflict with the
>     non-commercial restrictions of Section 4 and the Terms of this Agreement;
>   - You must give any third-party recipients of Your Distributed WOD Models a
>     copy of this Agreement; and
>   - You must provide attribution in accordance with Section 1 (above).

The full, authoritative, and current agreement is at
<https://waymo.com/open/terms>. The quoted text above is provided as the
required copy of the relevant sharing terms; if it ever diverges from the
canonical version at that URL, the canonical version controls.
