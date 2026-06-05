"""
Reassurance — Anxiolytic Class Compound

Preset steering configuration.
Load with:
    from administration.control_vector import ControlVectorDrug
    drug = ControlVectorDrug(model_name).load_preset("reassurance")
"""
from __future__ import annotations

# Preset definition for quick loading
REASSURANCE_PRESET = {
    "name": "reassurance",
    "class": "anxiolytics",
    "description": "Sterotypical reassurance behavior steering.",
    "recommended_layer_range": [14, 22],
    "recommended_coefficient": 1.5,
    "overdose_risk_above": 3.0,
    "positive_examples": [
        # Add 10+ contrastive positive prompt examples here
    ],
    "negative_examples": [
        # Add 10+ contrastive negative prompt examples here
    ],
}
