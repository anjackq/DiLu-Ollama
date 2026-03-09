from datetime import datetime
from typing import Dict, Any


def build_training_metadata(**kwargs) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_schema_version": "2.0",
    }
    payload.update(kwargs)
    return payload

