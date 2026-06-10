"""Tool system for the voice agent.

Gemma emits tool calls as lines like `[tool:name] optional args` inside its
reply. The agent parses them, runs the matching function, and (for tools that
return information, e.g. web search) does a second LM pass to turn the result
into a spoken answer.

This module is imported by the agent process (any TTS venv). It only uses the
standard library, so it adds no dependencies.

Adding a tool = one `@tool(...)`-decorated function. Its description is shown
to the model in the system prompt, so write it as an instruction.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional


# Matches a tool call line:  [tool:name] args...   (args optional, to EOL)
TOOL_RE = re.compile(r"\[tool:(\w+)\]\s*(.*?)(?:\n|$)", re.IGNORECASE)


@dataclass
class ToolResult:
    """What a tool returns to the agent."""

    followup: Optional[str] = None  # if set, feed this text back to the LM
    # for a 2nd pass (e.g. search results)
    terminal: bool = False  # if True, end the conversation (sleep)
    note: str = ""  # short log string
    animation: Optional[str] = None  # optional robot animation name to play
    timer_seconds: Optional[float] = None  # schedule a background timer
    timer_label: str = ""                  # spoken label for the timer


@dataclass
class Tool:
    name: str
    description: str  # shown to the model in the system prompt
    fn: Callable[[str, dict], ToolResult]


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str):
    def deco(fn: Callable[[str, dict], ToolResult]) -> Callable:
        TOOLS[name] = Tool(name=name, description=description, fn=fn)
        return fn

    return deco


# ---------------------------------------------------------------------------
# Parsing + prompt
# ---------------------------------------------------------------------------


def parse_tool_calls(text: str) -> tuple[list[tuple[str, str]], str]:
    """Return ([(name, args), ...], text_with_tool_lines_removed)."""
    calls: list[tuple[str, str]] = []

    def _repl(m: "re.Match") -> str:
        calls.append((m.group(1).lower(), m.group(2).strip()))
        return ""

    cleaned = TOOL_RE.sub(_repl, text).strip()
    return calls, cleaned


def tools_system_prompt() -> str:
    """Tool documentation appended to the system prompt."""
    lines = [
        "OUTILS:",
        "Tu peux déclencher une action avec une ligne au format : [tool:nom] arguments",
        "",
        "RÈGLES DE FORMAT (très important) :",
        "- Ta phrase parlée va sur sa PROPRE ligne.",
        "- La ligne [tool:...] va sur une ligne SÉPARÉE, en DERNIER.",
        "- Ne mets JAMAIS de phrase parlée après [tool:nom] sur la même ligne.",
        "- Pour [tool:search], l'argument est UNIQUEMENT la requête de recherche.",
        "",
        "Exemple correct (on te demande de dormir) :",
        "  [heard] tu peux aller dormir",
        "  D'accord, à bientôt !",
        "  [tool:sleep]",
        "",
        "Exemple correct (question d'actualité) :",
        "  [heard] quelle est la météo à Paris demain",
        "  Je vérifie ça tout de suite.",
        "  [tool:search] météo Paris demain",
        "",
        "Outils disponibles (n'en invente pas d'autres) :",
    ]
    for t in TOOLS.values():
        lines.append(f"  - {t.description}")
    lines.append("Si aucune action n'est nécessaire, ne mets aucune ligne [tool:...].")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@tool(
    "sleep",
    "[tool:sleep] — quand on te demande de dormir,  de te rendormir, de te taire, chut ou ta geule, ou"
    "d'arrêter d'écouter ou bien de faire dodo. Ou quand tu pense que la conversation est finie.",
)
def _sleep(args: str, ctx: dict) -> ToolResult:
    return ToolResult(terminal=True, note="sleep", animation="sleep")


@tool(
    "search",
    "[tool:search] <requête> — chercher une information à jour sur "
    "internet (météo, actualité, fait précis que tu ne connais pas).",
)
def _search(args: str, ctx: dict) -> ToolResult:
    query = args.strip()
    if not query:
        return ToolResult(
            followup="Recherche vide, demande une précision à l'utilisateur."
        )
    results = web_search(query)
    if results is None:
        return ToolResult(
            followup="La recherche internet n'est pas disponible "
            "(aucun moteur installé). Dis-le poliment."
        )
    if not results:
        return ToolResult(followup=f"Aucun résultat web pour «{query}». Dis-le.")
    txt = f"Résultats web pour «{query}» :\n"
    for i, r in enumerate(results, 1):
        txt += f"{i}. {r['title']} — {r['desc']}\n"
    txt += (
        "À partir de ces infos, réponds à l'utilisateur en français, "
        "en 1 ou 2 phrases courtes et naturelles."
    )
    return ToolResult(followup=txt, note=f"search:{query}")


@tool(
    "timer",
    "[tool:timer] <durée> — lancer un minuteur, ex. « 5 minutes », "
    "« 30 secondes », « 1h30 ». Tu seras prévenu à voix haute à la fin.",
)
def _timer(args: str, ctx: dict) -> ToolResult:
    secs, label = parse_duration(args)
    if not secs or secs <= 0:
        return ToolResult(
            followup="Je n'ai pas compris la durée du minuteur. Demande à "
                     "l'utilisateur de préciser (ex. « 5 minutes »)."
        )
    return ToolResult(timer_seconds=float(secs), timer_label=label,
                      note=f"timer:{secs}s")


_DURATION_UNITS = [
    (r"heures?|h", 3600),
    (r"minutes?|min|mn|m", 60),
    (r"secondes?|sec|s", 1),
]


def parse_duration(text: str) -> tuple[Optional[int], str]:
    """Parse a French duration string into (seconds, label).

    Handles « 5 minutes », « 30 secondes », « 1h30 », « 1 minute 30 »,
    « 90s », and a bare number (treated as seconds). Returns (None, text)
    if nothing parses.
    """
    t = (text or "").lower().strip()

    # "1h30" / "2 h 15" → hours + minutes
    m = re.search(r"(\d+)\s*h\s*(\d+)", t)
    if m:
        total = int(m.group(1)) * 3600 + int(m.group(2)) * 60
        return total, text.strip()

    # "1 minute 30" (trailing bare number = seconds), only if no explicit
    # seconds unit follows.
    m = re.search(r"(\d+)\s*(?:minutes?|min|mn)\s+(\d+)\s*$", t)
    if m:
        total = int(m.group(1)) * 60 + int(m.group(2))
        return total, text.strip()

    total = 0
    found = False
    for pattern, mult in _DURATION_UNITS:
        for num in re.findall(rf"(\d+)\s*(?:{pattern})\b", t):
            total += int(num) * mult
            found = True

    if not found:
        m = re.search(r"\d+", t)
        if m:
            return int(m.group(0)), text.strip()   # bare number → seconds
        return None, text.strip()

    return total, text.strip()


# --- placeholders (à implémenter plus tard) ---------------------------------


@tool(
    "emote",
    "[tool:emote] <émotion> — exprimer une émotion : happy, sad, "
    "surprised, curious, thinking.",
)
def _emote(args: str, ctx: dict) -> ToolResult:
    # TODO: antennes + micro-mouvements de tête selon l'émotion
    return ToolResult(note=f"emote:{args}", animation=f"emote:{args.strip()}")


@tool("look", "[tool:look] <direction> — tourner la tête : left, right, up, down.")
def _look(args: str, ctx: dict) -> ToolResult:
    # TODO: goto_target vers la direction
    return ToolResult(note=f"look:{args}", animation=f"look:{args.strip()}")


@tool("nod", "[tool:nod] — hocher la tête pour dire oui.")
def _nod(args: str, ctx: dict) -> ToolResult:
    # TODO
    return ToolResult(note="nod", animation="nod")


@tool("shake", "[tool:shake] — secouer la tête pour dire non.")
def _shake(args: str, ctx: dict) -> ToolResult:
    # TODO
    return ToolResult(note="shake", animation="shake")


@tool("dance", "[tool:dance] — faire une petite danse rigolote.")
def _dance(args: str, ctx: dict) -> ToolResult:
    # TODO
    return ToolResult(note="dance", animation="dance")


# ---------------------------------------------------------------------------
# Web search — DuckDuckGo by default (free, no key), Brave if a key is set
# ---------------------------------------------------------------------------


def web_search(query: str, count: int = 3) -> Optional[list[dict]]:
    """Return [{title, desc, url}], or None if no backend is available.

    Picks Brave when BRAVE_API_KEY is set (better quality), otherwise falls
    back to DuckDuckGo via the free `ddgs` package (no key, no signup).
    """
    if os.environ.get("BRAVE_API_KEY"):
        return _brave_search(query, count)
    return _ddg_search(query, count)


def _ddg_search(query: str, count: int = 3) -> Optional[list[dict]]:
    try:
        from ddgs import DDGS
    except ImportError:
        return None  # ddgs not installed
    try:
        rows = DDGS().text(query, max_results=count, region="fr-fr")
    except Exception as e:
        return [{"title": "erreur de recherche", "desc": str(e), "url": ""}]
    out = []
    for r in rows:
        out.append(
            {
                "title": _strip_html(r.get("title", "")),
                "desc": _strip_html(r.get("body", "")),
                "url": r.get("href", ""),
            }
        )
    return out


def _brave_search(query: str, count: int = 3) -> Optional[list[dict]]:
    """Brave Search API (needs a paid/verified key). Kept as an optional
    higher-quality backend when BRAVE_API_KEY is set."""
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return None
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": count}
    )
    req = urllib.request.Request(
        url,
        headers={
            "X-Subscription-Token": key,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except Exception as e:
        return [{"title": "erreur de recherche", "desc": str(e), "url": ""}]
    out = []
    for r in data.get("web", {}).get("results", [])[:count]:
        out.append(
            {
                "title": _strip_html(r.get("title", "")),
                "desc": _strip_html(r.get("description", "")),
                "url": r.get("url", ""),
            }
        )
    return out


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch(name: str, args: str, ctx: dict) -> ToolResult:
    t = TOOLS.get(name)
    if t is None:
        return ToolResult(note=f"outil inconnu: {name}")
    try:
        return t.fn(args, ctx)
    except Exception as e:
        return ToolResult(note=f"tool {name} a échoué: {type(e).__name__}: {e}")
