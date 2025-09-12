from typing import Dict


def compute_score(milestones_hit: set[str], rubric: Dict) -> Dict:
    """Computes the scenario score based on achieved milestones and a rubric.

    This function calculates scores for each dimension and the total score using milestone presence and rubric weights.

    Args:
        milestones_hit (set[str]): The set of milestone identifiers achieved.
        rubric (Dict): The scoring rubric containing dimension weights.

    Returns:
        Dict: A dictionary with scores by dimension and the total score.
    """
    # Simple: presence-based scores + room for future nuance
    weights = rubric["weights"]
    dims = {
        "evidence": 0,
        "reasoning": 0,
        "communication": 0,
        "discovery": (
            1.0
            if {"M1_goal", "M3_context"}.issubset(milestones_hit)
            else (
                0.5
                if "M1_goal" in milestones_hit or "M3_context" in milestones_hit
                else 0
            )
        ),
    }
    # Evidence: M2
    dims["evidence"] = 1.0 if "M2_logs" in milestones_hit else 0
    # Reasoning: M4
    dims["reasoning"] = 1.0 if "M4_solution" in milestones_hit else 0
    # Communication baseline; challenge injection can adjust (handled upstream)
    dims["communication"] = 1.0
    total = sum(dims[k] * weights[k] for k in weights)
    return {"by_dimension": dims, "total": round(total, 3)}
