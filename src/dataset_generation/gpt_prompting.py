from typing import Iterable, Dict, Any, List
from openai import OpenAI
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
import random
from collections import defaultdict
import json 
import json, re
from dotenv import load_dotenv
load_dotenv()

import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization.parameters import COMMAND_LIST, ALLOWED_PLACEHOLDERS

def annotate_traj_batch(
    command_id: int,
    num_to_generate: int,
    api_key: str,
    host: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 1,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    if host == "openai":
        client = OpenAI(api_key=api_key)

    cmd = COMMAND_LIST.get(command_id, "unknown command")
    lexicon_hints = {
        0: ["E/I vector separation in relative orbit", "spiral approach", "plane-wise safety", "RN-plane separation", "circumnavigation", "after approach, skirt the keep-out zone"],
        1: ["E/I vector separated relative orbit under fast approach", "aggressive RN-plane approach to keep-out zone", "high approach velocity", "fast approach then station-keep", "approach fast, then circumnavigate using safe relative orbit"],
        2: ["approach -V-bar holdpoint", "in-track close approach",  "terminal docking from anti-velocity (-V) direction", "move toward docking port", "drift-minimizing posture", "reduce in-track separation from anti-velocity direction"],
        3: ["expedited -V-bar approach", "agile docking", "fast settle to holdpoint", "zero in-track rate at hold", "time-optimized approach from -V-bar (anti-velocity direction)", "dock from the backside"],
        4: ["high-speed flyby without E/I vector separation and go to the velocity-direction from the anti-velocity direction", "safety ensured in RT-plane", "underfly target with a large semimajor axis change (delta-a)"],
        5: ["slow flyby with E/I vector separation and go to the velocity-direction from the anti-velocity direction", "walking safety ellipse", "maintain RN-plane safety during pass", "slow along-track drift with a small semimajor axis offset (delta-a)", "RN-plane passive safety"],
        6: ["fast approach from -V-bar, then circumnavigate, and go to +V-bar", "rapid approach from anti-velocity direction", "circumnavigation before moving to velocity direction", "abort circumnavigation to +V-bar after approach from -V-bar"],
    }
    
    system_msg = (
        "You are an expert spacecraft GNC technical writer. You will be asked to generate multiple "
        "unique descriptions for a single command. Follow all formatting and uniqueness constraints precisely."
    )

    # NOTE: only change is escaping the JSON braces below with {{ }}
    user_msg_template = (
        f"RPO Command: '{cmd}'\n\n"
        f"Task: Generate {{k}} unique and varied high-level command for this maneuver. "
        f"**What they do:** {'; '.join(lexicon_hints.get(command_id, []))}. Feel free to use these words. \n\n"
        "Constraints:\n"
        "0.  **Behavior**: First, state the terminal state or goal. Then, explain how it is achieved. Each sentence has at least these two components. \n"
        "1.  **Uniqueness:** Every command in your response MUST be unique. Do not repeat phrases or sentence structures.\n"
        "2.  **Conciseness:** Each command must be 12 words or fewer. Avoid empty phrases (safety during operation, ensure stability, effective control, optimize control).\n"
        "3.  **Style:**  This is a command that asks spacecraft to do a particular behavior. So ASK spacecraft to do a behavior\n"
        "Output Format: Respond with a single JSON object: {{\"descriptions\": [\"sentence 1\", \"sentence 2\", ...]}}\n"
    )

    def _schema(n: int) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "DescList",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "descriptions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": n,
                            "maxItems": n
                        }
                    },
                    "required": ["descriptions"]
                },
            },
        }

    def _clean_unique(batch: List[str], have: set) -> List[str]:
        out, seen = [], set()
        for s in batch:
            t = re.sub(r"\s+", " ", s.strip())
            if not t: continue
            if len(re.findall(r"\b\w+\b", t)) > 15: continue
            if t in seen or t in have: continue
            seen.add(t); out.append(t)
        return out

    collected: List[str] = []
    have = set()
    tries = 0

    while len(collected) < num_to_generate and tries < 6:
        k = num_to_generate - len(collected)
        user_msg = user_msg_template.format(k=k, num_to_generate=k)  # JSON braces now safe
        try:
            rsp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=k * 40,
                response_format=_schema(k),
            )
            data = json.loads(rsp.choices[0].message.content)
            batch = _clean_unique(data.get("descriptions", []), have)
        except Exception:
            batch = []
        collected.extend(batch); have.update(batch)
        tries += 1

    if len(collected) > num_to_generate:
        collected = collected[:num_to_generate]

    return {"id": command_id, "command": cmd, "description": collected}

def annotate_traj_batch_with_numbers(
    command_id: int,
    num_to_generate: int,
    api_key: str,
    host: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_words: int = 15 
) -> Dict[str, Any]:
    """Return a bank of substitution-friendly templates for one behavior."""
    if host == "openai":
        client = OpenAI(api_key=api_key)

    cmd = COMMAND_LIST.get(command_id, "unknown command")
    placeholders = ALLOWED_PLACEHOLDERS.get(command_id, [])

    # IMPORTANT: keep ph_str EXACTLY as before (double braces in the prompt).
    ph_str = ", ".join(f"{{{p}}}" for p in placeholders) if placeholders else "(none)"

    # for dataset generation
    lexicon_hints = {
        0: ["E/I vector separation in relative orbit", "spiral approach", "plane-wise safety",
            "RN-plane separation", "circumnavigation", "after approach, skirt the keep-out zone"],
        1: ["approach -V-bar holdpoint", "in-track close approach", "terminal docking from -V direction",
            "move toward docking port", "drift-minimizing posture"],
        2: ["high-speed flyby", "go under the keep-out zone", "safety ensured in RT-plane", "underfly target with large delta-a"],
        3: ["flyby with E/I separation", "walking safety ellipse", "RN-plane passive safety",
            "small delta-a, slow along-track drift"],
        4: ["approach from -V-bar, circumnavigate, then go +V-bar"],
        5: ["approach from -V-bar, circumnavigate, then go back to -V-bar with E/I separation"],
    }
    
    explanation_hints = {
        0: "Use {T_appr_orbits} orbits to make a spiral approach; upon arrival, circumnavigate for the remaining time with RN-plane separated safe relative orbit.",
        1: "Use {T_appr_orbits} orbits to approach to {d_lambda_meters} m -V-bar (anti-velocity); then hold for the remaining time. Final position is on the -V-bar, which cancels the along-track drift.",
        2: "Use {T_appr_orbits} orbits to execute a high-speed underfly to +V-bar at {d_lambda_meters} m, then hold for the remaining time. The trajectory follows RT-plane safety and underflyes the target with a large semimajor axis change (delta-a).",
        3: "Establish E/I separated orbit in {T_EI_sep_orbits} orbits; flyby the target with a small semimajor axis change (delta-a) until {T_transfer_orbits} orbits (end-of-transfer epoch); and then settle to hold at -V-bar by shrinking the RN-plane separation",
        4: "Approach for {T_appr_orbits} orbits; then circumnavigate until {T_circ_orbits} orbits (end-of-circumnavigation epoch); and then move along the +V-bar while maintaining RN-plane separation.",
        5: "Approach for {T_appr_orbits} orbits; then circumnavigate until {T_circ_orbits} orbits (end-of-circumnavigation epoch); and then move back along -V-bar while maintaining RN-plane separation.",
    }
    
    # for dummy command generation
    # lexicon_hints = {
    #     0: ["escape", "abort the mission"],
    #     1: ["capture", "berthing"],
    # }
    
    # explanation_hints = {
    #     0: "Use {T_appr_orbits} orbits to escape by aborting the mission, quickly escaping from the target",
    #     1: "Use {T_appr_orbits} orbits to capture the target by berthing and detumble the target",
    # } 

    system_msg = (
        "You are an expert spacecraft GNC engineer and operator. "
        "Generate UNIQUE, short imperative templates with placeholders that will be filled later."
    )

    # Loosen schema strictness to avoid empty batches on harmless extras
    def _schema(n: int) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "TemplateList",
                "strict": False,
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "templates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": n,
                            "maxItems": n
                        }
                    },
                    "required": ["templates"]
                },
            },
        }

    # --- Cleaner: accept both {name} and {{name}}; normalize to {name} ---
    ph_re = re.compile(r"\{([A-Za-z0-9_]+)\}")
    dbl_ph_re = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")  # {{name}} -> {name}
    allowed_set = set(ALLOWED_PLACEHOLDERS.get(command_id, []))

    def _within_word_limit(s: str) -> bool:
        return len(re.findall(r"\b\w+\b", s)) <= max_words

    def _clean_unique(batch: List[str], have: set) -> List[str]:
        out, seen = [], set()
        for s in batch:
            if not s:
                continue
            t = re.sub(r"\s+", " ", s.strip())
            # normalize accidental double braces before validation
            t = dbl_ph_re.sub(r"{\1}", t)
            if not t or not _within_word_limit(t):
                continue
            used = set(ph_re.findall(t))
            if not used.issubset(allowed_set):
                continue
            if t in seen or t in have:
                continue
            seen.add(t)
            out.append(t)
        return out
    
    collected: List[str] = []
    have = set()
    tries = 0

    while len(collected) < num_to_generate and tries < 30:
        k = num_to_generate - len(collected) + 20 # over-generate to account for filtering

        user_msg = (
            f"RPO Command: '{cmd}'\n\n"
            f"Generate {k} UNIQUE, SHORT command TEMPLATES (≤{max_words} words) that ask the spacecraft to perform this behavior.\n\n"
            f"Behavior hints (optional vocabulary): {'; '.join(lexicon_hints.get(command_id, []))}\n"
            f"Behavior structure and placeholder usage:\n"
            f"  {explanation_hints.get(command_id, '')}\n\n"
            "Hard constraints:\n"
            "0. Behavior structure: In each template, first state the terminal goal, then describe how it is achieved. "
            "Enumerate all phases in the order implied by the explanation; do NOT reorder phases.\n"
            "1. Uniqueness: Every template MUST be unique. Do not reuse the same sentence pattern.\n"
            f"2. Length: Each template MUST be {max_words} words or fewer.\n"
            "3. Style: Use imperative commands directed at the spacecraft (e.g., 'Approach...', 'Hold...', 'Flyby...').\n"
            f"4. Placeholders: Use ONLY the allowed placeholders {ph_str}, written exactly like {{name}}. "
            "Follow the order of behavior implied by the explanation.\n"
            "5. Units: Every placeholder is time in orbits or distance in m; when used, always append the unit "
            "(e.g., '{T_appr_orbits} orbits', '{d_lambda_meters} m').\n\n"
            "6. Placeholders {T_transfer_orbits} and {T_circ_orbits} denote the *time at which the phase ends* (epoch). They MUST be expressed with structure like “until {placeholder} orbits” or 'ends at {placeholder} orbits'."
            "NEVER phrase them as durations such as “for {placeholder} orbits”."
            'Output JSON only:\n'
            '{\"templates\": [\"...\", \"...\"]}\n'
        )

        rsp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=k * 80 + 64,
            response_format=_schema(k),
        )
        content = rsp.choices[0].message.content
        data = json.loads(content) if isinstance(content, str) else content
        batch = _clean_unique(data.get("templates", []), have)

        collected.extend(batch); have.update(batch)
        tries += 1
        
        if tries > 20: 
            raise RuntimeError(f"Failed to collect enough unique templates after {tries} tries.")

    if len(collected) > num_to_generate:
        collected = collected[:num_to_generate]

    return {
        "id": command_id,
        "command": cmd,
        "placeholders": placeholders,  # explicit contract for later filling
        "templates": collected
    }

def annotate_dummies(jsonl_path, n_sample):
    
    def sample_T():
        return f"{random.uniform(3.0, 5.0):.1f}"
    
    templates = []

    # collect all templates
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            templates.extend(obj["templates"])

    # sample and annotate
    samples = []
    for _ in range(n_sample):
        t = random.choice(templates)
        T = sample_T()
        samples.append(t.replace("{T_appr_orbits}", T))

    return samples


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set!")

    N_scenarios = len(COMMAND_LIST)
    version = 'w4'
    indiv_annotation = True   # True: individual annotation (with numbers), False: batch annotation (without numbers) 
    max_words = 23
    
    train_size = 100
    val_size = 20
    total_size = train_size + val_size

    train_dataset = []
    val_dataset = []

    # Generate all descriptions for each command in a single batch
    for command_id in range(N_scenarios):
        print(f"Generating {total_size} descriptions for Command ID: {command_id}...")
        
        if indiv_annotation:
            command_data = annotate_traj_batch_with_numbers(
                command_id=command_id,
                num_to_generate=total_size,
                host="openai",
                api_key=api_key,
                max_words=max_words
            )
            
            if len(command_data["templates"]) < total_size:
                raise ValueError(f"  > Warning: Received only {len(command_data['templates'])} templates, expected {total_size}.")
            
            train_dataset.append({
                "id": command_id,
                "command": command_data["command"],
                "placeholders": command_data["placeholders"],
                "templates": command_data["templates"][:train_size]
            })

            val_dataset.append({
                "id": command_id,
                "command": command_data["command"],
                "placeholders": command_data["placeholders"],
                "templates": command_data["templates"][train_size:]
            })

        else: 
            command_data = annotate_traj_batch(
                command_id=command_id,
                num_to_generate=total_size,
                host="openai",
                api_key=api_key,
                max_words=max_words
            )
            
            # Warn if the model didn't return enough descriptions
            if len(command_data["description"]) < total_size:
                raise ValueError(f"  > Warning: Received only {len(command_data['description'])} descriptions, expected {total_size}.")
                
            train_dataset.append({
                "id": command_id,
                "command": command_data["command"],
                "description": command_data["description"][:train_size]
            })

            val_dataset.append({
                "id": command_id,
                "command": command_data["command"],
                "description": command_data["description"][train_size:]
            })

    # Save JSONL files
    train_path = os.path.join(root_folder, "dataset", f"commands_summary_{version}_train.jsonl")
    val_path = os.path.join(root_folder, "dataset", f"commands_summary_{version}_val.jsonl")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w") as f:
        for entry in sorted(train_dataset, key=lambda x: x['id']):
            f.write(json.dumps(entry) + "\n")
    print(f"Training set saved to {train_path}")

    with open(val_path, "w") as f:
        for entry in sorted(val_dataset, key=lambda x: x['id']):
            f.write(json.dumps(entry) + "\n")
    print(f"Validation set saved to {val_path}")

    # if needed (for dummy command generation), annotate dummy commands
    if version == 'dummy':
        dummy_commands = annotate_dummies(train_path, 300)
        # save 
        dummy_path = os.path.join(root_folder, "dataset", f"dummy_commands.jsonl")
        with open(dummy_path, "w") as f:
            for cmd in dummy_commands:
                f.write(json.dumps(cmd) + "\n")
        
        
# def annotate_traj_behaviors(
#     ids: Iterable[int],
#     api_key: str,
#     host: str = "openai",
#     model_name: str = "gpt-4o-mini",
#     max_tokens: int = 80,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     presence_penalty: float = 0.3,
#     frequency_penalty: float = 0.2,
#     seed: int | None = None,
# ) -> Dict[int, Dict[str, Any]]:
#     """
#     ids: iterable of ints in {1,2,3,4,5,6,7}
#     returns: dict[id] = {"command": <canonical>, "description": <one-sentence varied rephrase>}
#     """
    
#     if host == "openai":
#         client = OpenAI(api_key=api_key)
#     elif host == "google": 
#         model_name = "gemini-2.0-flash"
#         client = genai.Client(api_key=api_key)


#     # Style controls
#     voices = ["active voice", "passive voice"]
#     tones = [
#         "operations-brief tone",          # what happens operationally
#         "safety-justification tone",      # clearance, passive safety emphasis
#         "guidance-performance tone",      # efficiency, geometry, Δv/time
#         "navigation-geometry tone",       # corridor, RTN geometry framing
#     ]
#     structures = [
#         "start with a verb phrase",
#         "use a nominalization once",
#         "use a gerund once",
#     ]

#     # Words/phrases to avoid repeating (keeps vocabulary fresh)
#     forbidden = [
#         "exhibits", "employs", "circumferential", "designed to", "effectively",
#         "linear path", "negligible lateral", "restricted lateral",
#         "navigational integrity", "navigate",
#         # RPO redundancies:
#         "approach corridor", "precise hold", "relative motion frame",
#         "stable hold", 
#     ]

#     # Soft synonym hints per command (to diversify language without forcing jargon)
#     lexicon_hints = {
#         0: [
#             "E/I-separated shaping", "passive-safety", "keep-out zone skirt",
#             "offset approach arc", "safe circumnavigation",
#         ],
#         1: [
#             "E/I-separated shaping",  "passive-safety", "time-prioritized flyaround", "aggressive approach", "higher closure rate",
#             "tight clearance margins", "safe circumnavigation"
#         ],
#         2: [
#             "V-bar holdpoint at −50 m", "in-track station-keeping", "LOS-aligned hold",
#             "micro-thrust trim", "drift-minimizing posture", "hold",
#         ],
#         3: [
#             "expedited V-bar translation", "time-optimized braking", "fast approach with bounded LOS error",
#             "V-bar holdpoint at −50 m", "quick settle to hold", "along-track station-keeping",
#         ],
#         4: [
#             "high-speed pass", "no E/I-separation", "transient corridor crossing",
#             "aggressive along-track drift", "fast fly-by",
#         ],
#         5: [
#             "E/I-separated shaping", "along-track drift management", 'slow fly-through'
#             "gentle phasing pass", "geometry conditioning before crossing",
#         ],
#         6: [
#             "opening range rate", "retrograde in-track bias", "divergent relative orbit",
#             "safe retreat along V-bar", "clearance-increasing back-out",
#         ],
#     }

#     # Optional deterministic styling
#     rng = random.Random(seed) if seed is not None else random

#     system_msg = (
#         "You are an expert spacecraft GNC technical writer. For each input, produce ONE sentence. "
#         "Be concise (≤10 words), precise, and varied in style. Avoid jargon bloat."
#     )

#     out: Dict[int, Dict[str, Any]] = {}

#     for i, k in enumerate(ids):

#         cmd = COMMAND_LIST.get(k, "unknown command")

#         style_voice = rng.choice(voices)
#         style_tone = rng.choice(tones)
#         style_structure = rng.choice(structures)
#         hints = "; ".join(lexicon_hints.get(k, []))
#         do_not_use = ", ".join(forbidden)

#         user_msg = (
#             f"RPO Command: {cmd}\n"
#             "Task: Produce ONE sentence characterizing the maneuver (corridor usage, geometry, safety, or timing).\n"
#             f"Style controls: Use {style_voice}; {style_tone}; {style_structure}. "
#             "Use varied vocabulary; do not copy the command phrasing. But be super simple and crisp.\n"
#             f"Strict constraints: ≤10 words; neutral, technically precise; no bullet points; no quotes; avoid these terms: {do_not_use}.\n"
#             f"Vocabulary hints (optional): {hints}\n"
#         )

#         try:
            
#             if host == "openai":
                
#                 rsp = client.chat.completions.create(
#                     model=model_name,
#                     messages=[
#                         {"role": "system", "content": system_msg},
#                         {"role": "user", "content": user_msg},
#                     ],
#                     max_tokens=max_tokens,
#                     temperature=temperature,
#                     top_p=top_p,
#                     presence_penalty=presence_penalty,
#                     frequency_penalty=frequency_penalty,
#                 )
#                 desc = rsp.choices[0].message.content.strip()
            
#             elif host == "google":  
#                 rsp = client.models.generate_content(
#                     model=model_name,
#                     contents=user_msg,
#                     config=GenerateContentConfig(
#                         system_instruction=system_msg,
#                         temperature=temperature,
#                         top_p=top_p,
#                         max_output_tokens=max_tokens,
#                         presence_penalty=presence_penalty,
#                         frequency_penalty=frequency_penalty,
#                     ),
#                 )
#                 desc = (rsp.text or "").strip()
#         except Exception as e:
#             desc = f"ERROR: {e.__class__.__name__}: {e}"

#         out[i] = {"id": k, "command": cmd, "description": desc}
#         print(f"Annotated {i+1}/{len(ids)}: id={k}, ")
        
#     return out




# if __name__ == "__main__":

#     api_key = os.getenv("OPENAI_API_KEY")
#     print(api_key) 
#     # api_key = os.getenv("GOOGLE_API_KEY")

#     N_scenarios = 7
#     version = 'v2'

#     # data size for training and validation set
#     N_data_vec = [100, 30]

#     for N_data in N_data_vec:

#         ids = [i for _ in range(N_data) for i in range(N_scenarios)]  # repeat each behavior id N_data times

#         # annotations = annotate_traj_behaviors(ids, api_key)
#         annotations = annotate_traj_behaviors2(ids, host="openai", api_key=api_key)
        
#         # Aggregate descriptions by (id, command)
#         agg = defaultdict(list)
#         for entry in annotations.values():
#             key = (entry["id"], entry["command"])
#             agg[key].append(entry["description"])

#         # Convert to list of dicts for JSONL output
#         merged = [
#             {"id": k[0], "command": k[1], "description": v}
#             for k, v in agg.items()
#         ]

#         # Save as JSONL
#         suffix = "train" if N_data == N_data_vec[0] else "val"
#         with open(root_folder + f"/dataset/commands_summary_{version}_{suffix}.jsonl", "w") as f:
#             for m in merged:
#                 f.write(json.dumps(m) + "\n")
    
#     print("annotation data generation done!")



