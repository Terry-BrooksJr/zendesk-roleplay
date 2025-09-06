import re
from typing import Dict
from sentence_transformers import SentenceTransformer, util

_rules = {
  "ask_for_logs": [r"\bhar\b", r"network log", r"devtools", r"preserve log", r"export .*har"],
  "ask_for_goal": [r"\bgoal\b", r"what.*trying.*(do|achieve)", r"outcome"],
  "ask_for_repro": [r"repro(duce|duction)|steps|how.*reproduce"],
  "ask_for_context": [r"device|ipad|ios|version|browser|safari|schoology app|webview|file\s?(type|format)"],
  "propose_fix": [r"try|suggest|workaround|fix|solution|change|switch|configure|encode"]
}
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # small, fast

_semantic_prompts = {
  "ask_for_logs": "please share the HAR or network logs from devtools",
  "ask_for_goal": "what are you trying to achieve overall",
  "ask_for_repro": "please provide steps to reproduce the issue",
  "ask_for_context": "ask device os browser app webview and file-type details",
  "propose_fix": "propose a likely resolution or workaround"
}

def detect(text:str)->Dict[str,bool]:
    t = text.lower()
    hits = {k: False for k in _rules}
    # rules pass
    for name, pats in _rules.items():
        if any(re.search(p, t) for p in pats):
            hits[name] = True
    # semantic fallback
    emb = _model.encode([text], convert_to_tensor=True)
    for name, prompt in _semantic_prompts.items():
        if hits[name]: continue
        sim = util.cos_sim(emb, _model.encode([prompt], convert_to_tensor=True))[0][0].item()
        if sim > 0.62: hits[name] = True
    return hits