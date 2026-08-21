"""CUE demo: read a conversation, write a persona manual, then steer a user simulator with it.

The comparison tab is the point: both arms get the same task and the same assistant, and
only the CUE arm gets the manual, so any difference in how the user behaves comes from CUE.
"""

from __future__ import annotations

import os

import gradio as gr
import spaces
import torch
from cue_hf import CueModel, CueProcessor
from cue_hf.example_pool import inject_examples
from cue_hf.render import render_dual_manual
from cue_hf.schema import merge_dual_manual_blocks
from huggingface_hub import InferenceClient

CUE_REPO = os.environ.get("CUE_REPO", "handshake-ai-research/cue")
SIM_MODEL = os.environ.get("SIM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
POOL_REPO = os.environ.get("POOL_REPO", "handshake-ai-research/cue-example-pool")
TOKEN = os.environ.get("HF_TOKEN")
SLOT_MAX_TOKENS = 64  # slot commands are one-liners
# CUE itself runs on the Space's own GPU, but the simulator and the assistant are billed
# Inference Provider calls, so tab 2 is opt-in unless this is set for a screenshot run.
PAID_DEFAULT = os.environ.get("PAID_INFERENCE", "0") == "1"

# Loaded on CPU: ZeroGPU forbids touching CUDA outside a @spaces.GPU call.
MODEL = CueModel.from_pretrained(CUE_REPO, device="cpu", token=TOKEN).eval()
PROCESSOR = CueProcessor(session_preprocess="full")
CLIENT = InferenceClient(model=SIM_MODEL, token=TOKEN)

PRESETS = {
    "Terse editor (blunt, keeps pushing back)": """user: can you tighten this paragraph
assistant: Sure - here is a shorter version.
user: still too long, cut it in half
assistant: Trimmed to two sentences.
user: fine. now make the tone less stiff""",
    "Budget traveler (price-first, walks away)": """user: i need a flight to boston next friday, cheapest option
assistant: I found a 6am connecting flight for $180.
user: whats the price if i leave saturday instead
assistant: Saturday is $210 nonstop.
user: no thanks, too much. keep looking under 200""",
    "Careful learner (polite, asks why)": """user: Could you explain why my test is failing? Traceback attached.
assistant: The fixture returns None because the patch target is wrong.
user: Thank you! Which line should I change, and why does patching there help?
assistant: Line 14 - patch where the name is looked up, not where it is defined.
user: That makes sense. Is there a general rule I should remember for next time?""",
    "Chatty planner (informal, thinks out loud)": """user: hey so im trying to pick between two apartments
assistant: Happy to help - what matters most to you?
user: commute mostly!! but also i have a dog so yard matters
assistant: Then the second one, with the fenced yard.
user: yeah i was leaning that way too, thanks :)""",
}

SCENARIOS = [
    "You want help planning a 3-day trip to Lisbon on a tight budget.",
    "You want the assistant to rewrite your cover letter for a data analyst job.",
    "Your laptop will not connect to wifi and you want it fixed.",
    "You want to choose between two health insurance plans.",
]

USER_SIM_SYSTEM = (
    "You are role-playing a human user talking to an AI assistant. Stay in character, "
    "send exactly one short message per turn, and never speak or act as the assistant.\n\n"
    "Your goal in this conversation:\n{scenario}"
)
ASSISTANT_SYSTEM = "You are a helpful AI assistant. Answer concisely."


def parse_transcript(text: str) -> list[dict[str, str]]:
    """Read ``role: content`` lines, treating unlabeled lines as continuations."""

    turns: list[dict[str, str]] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        role, _, content = stripped.partition(":")
        if role.strip().lower() in {"user", "assistant", "system"} and content.strip():
            turns.append({"role": role.strip().lower(), "content": content.strip()})
        elif turns:
            turns[-1]["content"] += " " + stripped
        else:
            turns.append({"role": "user", "content": stripped})
    if not turns:
        raise gr.Error("Add at least one turn, like `user: can you shorten this`.")
    return turns


def _bullets(title: str, items: list[str]) -> str:
    if not items:
        return f"**{title}**\n\n_none_\n"
    return f"**{title}**\n\n" + "\n".join(f"- {item}" for item in items) + "\n"


@spaces.GPU(duration=180)
def extract_manual(transcript: str, preprocess: str, retrieval: bool):
    """Encode the conversation and decode one command per slot for each of the three heads."""

    sessions = PROCESSOR(parse_transcript(transcript))
    MODEL.to("cuda")
    with torch.no_grad():
        bottleneck = MODEL.encode(sessions, session_preprocess=preprocess)
        blocks = {
            head: MODEL.generate_command_slot_blocks(
                bottleneck,
                head=head,
                num_slots=num_slots,
                max_new_tokens=SLOT_MAX_TOKENS,
                temperature=0.0,
            )[0]
            for head, num_slots in (
                ("general", MODEL.config.general_command_slots),
                ("specific", MODEL.config.user_specific_command_slots),
                ("style", MODEL.config.style_command_slots),
            )
        }

    general, specific, style = blocks["general"], blocks["specific"], blocks["style"]
    merged = merge_dual_manual_blocks(general, specific, style)
    if retrieval:
        if MODEL.example_pool is None:
            MODEL.attach_example_pool(POOL_REPO)
        merged = inject_examples(
            merged,
            pool=MODEL.example_pool,
            cue_embedding=bottleneck[0].detach().float().cpu().numpy(),
            k_sessions=MODEL.config.example_retrieval_k_sessions,
            n_general=MODEL.config.example_retrieval_n_general,
            n_specific=MODEL.config.example_retrieval_n_specific,
        )
        specific = {**specific, "examples": merged.get("examples") or []}
        style = {**style, "examples": []}

    if not merged.get("commands"):
        raise gr.Error("The decoder returned no commands for this conversation - try a longer one.")

    view = "\n".join(
        [
            _bullets("General behavior", general.get("commands") or []),
            _bullets("User-specific behavior", specific.get("commands") or []),
            _bullets("Writing style", style.get("commands") or []),
            _bullets(
                "Style samples (retrieved)" if retrieval else "Style samples (decoded)",
                merged.get("examples") or [],
            ),
        ]
    )
    return view, render_dual_manual(general, specific, style)


def _chat(convo: list[dict[str, str]], system: str) -> str:
    reply = CLIENT.chat_completion(
        messages=[{"role": "system", "content": system}, *convo],
        max_tokens=160,
        temperature=0.8,
    )
    return (reply.choices[0].message.content or "").strip()


def _as_user_view(convo: list[dict[str, str]]) -> list[dict[str, str]]:
    """Flip roles so the simulator sees its own past messages as its own."""

    return [{"role": "assistant" if turn["role"] == "user" else "user", "content": turn["content"]} for turn in convo]


def compare(scenario: str, steering: str, turns: int, billing_ok: bool):
    """Run the same scenario twice - once with the manual, once without - turn by turn."""

    if not billing_ok:
        raise gr.Error("This tab spends Inference Provider credits (4 calls per turn). Tick the box to run it.")
    if not (scenario or "").strip():
        raise gr.Error("Pick or write a scenario first.")
    if not (steering or "").strip():
        raise gr.Error("Extract a manual on the first tab before comparing.")

    base_system = USER_SIM_SYSTEM.format(scenario=scenario.strip())
    arms = {
        "steered": {"convo": [], "system": f"{base_system}\n\nBehave like this specific user:\n{steering}"},
        "baseline": {"convo": [], "system": base_system},
    }
    for _ in range(int(turns)):
        for arm in arms.values():
            convo = arm["convo"]
            convo.append({"role": "user", "content": _chat(_as_user_view(convo), arm["system"])})
            convo.append({"role": "assistant", "content": _chat(convo, ASSISTANT_SYSTEM)})
        yield arms["steered"]["convo"], arms["baseline"]["convo"]


@spaces.GPU(duration=180)
def sample_users(n: int, seed: int):
    """Draw users from the diffusion prior - no input conversation required."""

    if MODEL.sampler is None:
        raise gr.Error(f"{CUE_REPO} has no sampler.pt and its config sets no sampler_id.")
    MODEL.to("cuda")
    out = MODEL.sample_user(n=int(n), seed=int(seed), max_new_tokens=SLOT_MAX_TOKENS)
    sections = []
    for index, manual in enumerate(out["manuals"], start=1):
        commands = (manual or {}).get("commands") or []
        sections.append(_bullets(f"Synthetic user {index}", commands))
    return "\n".join(sections) or "_the prior produced no commands; try another seed_"


with gr.Blocks(title="CUE - persona manuals for user simulation") as demo:
    gr.Markdown(
        "# CUE\n"
        "CUE reads a user's dialogue history and writes a **persona manual** - behavioral "
        "commands you can hand to any assistant LM so it role-plays that user.\n\n"
        f"Model: `{CUE_REPO}` · simulator: `{SIM_MODEL}`"
    )

    with gr.Tab("1. Conversation to manual"):
        with gr.Row():
            with gr.Column():
                first_preset = next(iter(PRESETS))
                preset = gr.Dropdown(list(PRESETS), value=first_preset, label="Preset conversation")
                transcript = gr.Textbox(
                    value=PRESETS[first_preset],
                    lines=12,
                    label="Dialogue history (`role: content` per line)",
                )
                preprocess = gr.Radio(
                    ["full", "strip_document", "user_only"],
                    value="full",
                    label="Session preprocessing",
                    info="Use strip_document when an assistant turn carries a whole draft.",
                )
                retrieval = gr.Checkbox(
                    value=False,
                    label="Retrieve style samples from the example pool",
                    info=f"Downloads {POOL_REPO} (~1.5 GB) the first time.",
                )
                run_extract = gr.Button("Extract manual", variant="primary")
            with gr.Column():
                gr.Markdown("### Manual")
                manual_view = gr.Markdown()
                steering_view = gr.Textbox(
                    label="Rendered steering prompt (what the simulator receives)",
                    lines=10,
                    info="Editable: change a command here and rerun tab 2 to see it land.",
                )
        preset.change(lambda name: PRESETS[name], preset, transcript)
        run_extract.click(extract_manual, [transcript, preprocess, retrieval], [manual_view, steering_view])

    with gr.Tab("2. Does the manual steer a simulator?"):
        gr.Markdown(
            "Same task, same assistant, same simulator - the only difference is whether the "
            "user side gets the manual from tab 1.\n\n"
            f"**This is the only tab that costs anything.** CUE runs on this Space's own GPU, "
            f"but both the simulated user and the assistant are `{SIM_MODEL}` calls through "
            "Inference Providers: 4 calls per turn, so a 3-turn run is 12."
        )
        with gr.Row():
            scenario = gr.Dropdown(SCENARIOS, value=SCENARIOS[0], allow_custom_value=True, label="Task for the user")
            turns = gr.Slider(1, 5, value=3, step=1, label="Turns")
            billing_ok = gr.Checkbox(value=PAID_DEFAULT, label="Spend inference credits")
            run_compare = gr.Button("Run both", variant="primary")
        with gr.Row():
            steered = gr.Chatbot(label="CUE-steered user", height=420)
            baseline = gr.Chatbot(label="Generic user (no manual)", height=420)
        run_compare.click(compare, [scenario, steering_view, turns, billing_ok], [steered, baseline])

    with gr.Tab("3. Sample users from the prior"):
        gr.Markdown(
            "The diffusion prior generates CUE embeddings with no input conversation, so you "
            "can populate an evaluation with synthetic users."
        )
        with gr.Row():
            count = gr.Slider(1, 6, value=3, step=1, label="Users")
            seed = gr.Number(value=0, precision=0, label="Seed")
            run_sample = gr.Button("Sample", variant="primary")
        sampled = gr.Markdown()
        run_sample.click(sample_users, [count, seed], sampled)


if __name__ == "__main__":
    demo.queue().launch()
