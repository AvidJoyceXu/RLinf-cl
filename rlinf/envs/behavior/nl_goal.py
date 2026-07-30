"""Render a BEHAVIOR/BDDL goal as a natural-language instruction.

This is the module that turns the benchmark from *predicate translation* into
*task planning*. With ground BDDL atoms in the prompt --

    inside(can__of__soda.n.01_1, ashcan.n.01_1)
    inside(can__of__soda.n.01_2, ashcan.n.01_1)
    inside(can__of__soda.n.01_3, ashcan.n.01_1)

-- the policy can reach the goal by transliterating each atom into the tool whose
name shares its semantics (``inside`` -> ``place_inside``), object names included.
What is measured is a lookup. With the same goal as natural language --

    "Every can of soda is inside the trash can."

-- the policy has to decide *which* predicates realise the sentence, *which*
objects in the scene the noun phrases denote, and in *what order* the tools must
fire. That is the research object (see
``brainstorm/260718 - research question, novelty & method - Claude.md``: NL goals
are the highest-priority change of the three).

Two properties are load-bearing:

* **The renderer reads the UNGROUND goal.** ``problem0.bddl``'s ``:goal`` keeps the
  quantifier structure (``forall``/``exists``/``forpairs``) that grounding expands
  away. Quantifiers are what make the sentence short and human ("every can of
  soda", not three indexed atoms), and they withhold the instance enumeration a
  ground atom list hands over for free.
* **Reward is untouched.** Only the prompt changes. Success still comes from the
  simulator's real ``BehaviorTask.goal_status`` over the ground atoms, so making
  the prompt vaguer cannot make the task easier to *score* -- only harder to
  *solve*. That asymmetry is the point, and it is why this is not a way of
  gaming the metric.

Stdlib only, and no OmniGibson: the s-expression scanner below is ~20 lines, which
is cheaper than making the py3.11 trainer venv carry ``bddl``. ``bddl`` is imported
only to locate ``activity_definitions/`` when no explicit path is given.

    # render one
    python -m rlinf.envs.behavior.nl_goal picking_up_trash
    # render all 1016 and report which constructs fell back
    python -m rlinf.envs.behavior.nl_goal --audit
"""
from __future__ import annotations

import os
import re
from typing import Any

Term = Any  # str | list[Term]


# --------------------------------------------------------------------------- #
# BDDL s-expression scanner
# --------------------------------------------------------------------------- #
def scan(text: str) -> list:
    """Parse a BDDL file body into nested lists. Mirrors bddl.parsing.scan_tokens."""
    text = re.sub(r";.*$", "", text, flags=re.MULTILINE).lower()
    stack: list[list] = []
    out: list = []
    for tok in re.findall(r"[()]|[^\s()]+", text):
        if tok == "(":
            stack.append(out)
            out = []
        elif tok == ")":
            if not stack:
                raise ValueError("unbalanced ')'")
            inner, out = out, stack.pop()
            out.append(inner)
        else:
            out.append(tok)
    if stack:
        raise ValueError("unbalanced '('")
    if len(out) != 1:
        raise ValueError("malformed BDDL")
    return out[0]


def activity_definitions_dir() -> str:
    """Locate bddl's activity_definitions/ (env var wins, then the package)."""
    env = os.environ.get("BDDL_ACTIVITY_DEFINITIONS")
    if env:
        return env
    import bddl

    return os.path.join(os.path.dirname(bddl.__file__), "activity_definitions")


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #
# Every predicate that occurs in any BEHAVIOR-1K :goal block, measured over all
# 1016 problem0.bddl files (counts as of bddl 3.7.0):
#   inside 781  covered 713  ontop 557  real 363  contains 248  nextto 166
#   cooked 116  attached 67  folded 60  filled 55  open 36  draped 30
#   overlaid 25  touching 22  frozen 19  hot 17  saturated 14  toggled_on 12
#   under 8  broken 3  unfolded 3  on_fire 2
# The negative form is written out rather than composed with "not", because
# English has a positive word for most of these ("closed", not "not open") and
# because "the floor is not covered with mud" reads as a weaker claim than the
# BDDL means.
POSITIVE = {
    "inside":     "{0} is inside {1}",
    "ontop":      "{0} is on top of {1}",
    "under":      "{0} is under {1}",
    "nextto":     "{0} is next to {1}",
    "touching":   "{0} is touching {1}",
    "attached":   "{0} is attached to {1}",
    "draped":     "{0} is draped over {1}",
    "overlaid":   "{0} is laid out flat over {1}",
    "covered":    "{0} is covered with {1}",
    "contains":   "{0} contains {1}",
    "filled":     "{0} is filled with {1}",
    "saturated":  "{0} is soaked with {1}",
    "cooked":     "{0} is cooked",
    "frozen":     "{0} is frozen",
    "hot":        "{0} is hot",
    "on_fire":    "{0} is on fire",
    "open":       "{0} is open",
    "toggled_on": "{0} is switched on",
    "folded":     "{0} is folded up",
    "unfolded":   "{0} is unfolded",
    "broken":     "{0} is broken",
}
NEGATIVE = {
    "inside":     "{0} is not inside {1}",
    "ontop":      "{0} is not on {1}",
    "under":      "{0} is not under {1}",
    "nextto":     "{0} is away from {1}",
    "touching":   "{0} is not touching {1}",
    "attached":   "{0} is detached from {1}",
    "draped":     "{0} is not draped over {1}",
    "overlaid":   "{0} is not laid over {1}",
    "covered":    "{0} has no {1} on it",
    "contains":   "{0} holds no {1}",
    "filled":     "{0} is empty of {1}",
    "saturated":  "{0} is not soaked with {1}",
    "cooked":     "{0} is raw",
    "frozen":     "{0} is not frozen",
    "hot":        "{0} is not hot",
    "on_fire":    "{0} is not burning",
    "open":       "{0} is closed",
    "toggled_on": "{0} is switched off",
    "folded":     "{0} is not folded",
    "unfolded":   "{0} is still folded",
    "broken":     "{0} is intact",
}

# `real` is existence, not a property of a known object, so it wants an
# existential sentence ("there are eight half logs") rather than the
# "<the X> is <predicate>" frame every other predicate uses. Handled in _leaf.
EXISTENTIAL = "real"

# Argument positions holding a particle/substance system rather than a rigid
# object. These are mass nouns in English -- "covered with mud", never "covered
# with the mud" -- and they are also not things `observe` reports as objects.
SYSTEM_ARG = {"covered": 2, "contains": 2, "filled": 2, "saturated": 2}

# Predicates whose two arguments play the same role, so `p(x_1, x_2)` over two
# instances of one category is idiomatically "the two Xs are ... each other"
# rather than two indistinguishable indexed names.
RECIPROCAL = {"nextto": "next to each other", "touching": "touching each other"}

ORDINAL = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
           "eighth", "ninth", "tenth"]

# Nouns whose singular form is already plural-shaped, so the templates' "is" has
# to become "are". Only the ones that occur as BEHAVIOR object categories.
PLURAL_FORM = {"pliers", "scissors", "tongs", "shears", "trousers", "jeans",
               "glasses", "goggles", "pants", "binoculars", "tweezers"}


def is_plural_subject(phrase: str) -> bool:
    """Does this noun phrase take a plural verb?

    True for aggregated counts ("eight half logs", "exactly two plates", "the two
    sandals") and for plural-shaped singulars ("the pliers"); false for "every X"
    and "at least one X", which are grammatically singular."""
    words = phrase.split()
    if not words:
        return False
    if words[0] in ("every", "each") or phrase.startswith("at least one "):
        return False
    for w in words:
        if w in NUMBER[2:] or w.isdigit():
            return True
    head = words[1] if words[0] == "the" and len(words) > 1 else words[0]
    return head in PLURAL_FORM

# Synsets whose lexicalisation is technically correct but not what a person says.
# Mass nouns get a unit word, because BDDL counts them as discrete instances and
# "every bacon" is not English while "every strip of bacon" is.
PREFERRED_NOUN = {
    "ashcan": "trash can",
    "electric refrigerator": "refrigerator",
    "gym shoe": "sneaker",
    "hallstand": "shoe rack",
    "bacon": "strip of bacon",
    "plywood": "sheet of plywood",
}

NUMBER = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def numword(n: int) -> str:
    return NUMBER[n] if n < len(NUMBER) else str(n)


def lexicalize(term: str) -> str:
    """``?can__of__soda.n.01_3`` -> ``can of soda``.

    BEHAVIOR encodes word boundaries as underscores and multi-word heads as
    double underscores, so both collapse to spaces; ``.n.01`` is the WordNet
    sense tag and ``_3`` the instance index, and neither is language."""
    t = term.lstrip("?")
    t = re.sub(r"\.[a-z]\.\d+.*$", "", t)      # drop .n.01 and any _<idx> after it
    t = re.sub(r"_\d+$", "", t)                # ...or a bare trailing _<idx>
    t = re.sub(r"_+", " ", t).strip()
    return PREFERRED_NOUN.get(t, t)


def pluralize(noun: str) -> str:
    """Pluralise the head of a noun phrase: ``box of oatmeal`` -> ``boxes of oatmeal``."""
    head, sep, tail = noun.partition(" of ")
    if re.search(r"(s|x|z|ch|sh)$", head):
        head += "es"
    elif re.search(r"[^aeiou]y$", head):
        head = head[:-1] + "ies"
    else:
        head += "s"
    return head + sep + tail


def article(noun: str) -> str:
    return "an " if noun[:1] in "aeiou" else "a "


# --------------------------------------------------------------------------- #
# Problem file -> goal AST + naming context
# --------------------------------------------------------------------------- #
class Problem:
    """The parts of a ``problem0.bddl`` the renderer needs.

    ``rooms`` comes from the ``:init`` ``inroom`` facts and is used *only* to
    disambiguate two instances of the same category ("the kitchen floor" vs "the
    living room floor"). That is what a person would say, and it is goal
    specification rather than a plan: it names the target, not the route or the
    tool. Set ``room_qualified=False`` to measure whether it matters."""

    def __init__(self, activity: str, definition_id: int = 0, bddl_dir: str | None = None):
        path = os.path.join(bddl_dir or activity_definitions_dir(),
                            activity, f"problem{definition_id}.bddl")
        with open(path) as f:
            tree = scan(f.read())

        self.activity = activity
        self.goal: Term | None = None
        self.type_of: dict[str, str] = {}       # instance -> synset type
        self.count: dict[str, int] = {}         # synset type -> #instances
        self.rooms: dict[str, str] = {}         # instance -> room

        for block in tree:
            if not isinstance(block, list) or not block:
                continue
            if block[0] == ":objects":
                self._read_objects(block[1:])
            elif block[0] == ":init":
                for atom in block[1:]:
                    if isinstance(atom, list) and len(atom) == 3 and atom[0] == "inroom":
                        self.rooms[atom[1].lstrip("?")] = atom[2]
            elif block[0] == ":goal":
                self.goal = block[1]

        if self.goal is None:
            raise ValueError(f"{activity}: no :goal block")

    def _read_objects(self, tokens: list[str]) -> None:
        """``a_1 a_2 - a  b_1 - b`` — instances, then '-', then their type."""
        pending: list[str] = []
        expect_type = False
        for tok in tokens:
            if tok == "-":
                expect_type = True
            elif expect_type:
                for inst in pending:
                    self.type_of[inst] = tok
                self.count[tok] = self.count.get(tok, 0) + len(pending)
                pending, expect_type = [], False
            else:
                pending.append(tok.lstrip("?"))


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
QUANT = {"forall", "exists", "forn", "forpairs", "fornpairs"}


class GoalRenderer:
    def __init__(self, problem: Problem, room_qualified: bool = True):
        self.p = problem
        self.room_qualified = room_qualified
        self.fallbacks: list[str] = []          # constructs that hit a generic path

    # -- terms ------------------------------------------------------------- #
    def noun(self, tok: str, bound: dict[str, str], bare: bool = False) -> str:
        if tok in bound:
            return bound[tok]
        inst = tok.lstrip("?")
        if inst.startswith("agent."):
            return "you"
        noun = lexicalize(inst)
        if bare:                                 # mass noun: "covered with mud"
            return noun
        # Ambiguous only when the problem declares several of this category AND
        # the goal could mean a specific one. A room adjective is the natural fix
        # ("the kitchen floor"); an ordinal is the fallback for genuinely
        # interchangeable items ("the first storage container"), which at least
        # keeps identity stable across clauses the way an index would.
        typ = self.p.type_of.get(inst)
        if typ and self.p.count.get(typ, 1) > 1:
            room = self.p.rooms.get(inst) if self.room_qualified else None
            if room:
                return f"the {room.replace('_', ' ')} {noun}"
            idx = inst.rsplit("_", 1)[-1]
            self.fallbacks.append(f"ambiguous instance {inst}")
            ord_ = ORDINAL[int(idx)] if idx.isdigit() and int(idx) < len(ORDINAL) else idx
            return f"the {ord_} {noun}"
        return f"the {noun}"

    # -- leaves ------------------------------------------------------------ #
    def _leaf(self, term: list, bound: dict[str, str], neg: bool) -> str:
        pred, args = term[0], term[1:]

        if pred == EXISTENTIAL:
            noun = self.noun(args[0], bound)
            bare = noun[4:] if noun.startswith("the ") else noun
            if neg:
                return f"no {bare} is left"
            # "there is a pizza" / (aggregated) "there are eight half logs"
            counted = bare.split(" ", 1)[0] in NUMBER or bare[:1].isdigit()
            return f"there are {bare}" if counted else f"there is {article(bare)}{bare}"

        # p(x_1, x_2) over one category: "the two sandals are next to each other".
        if pred in RECIPROCAL and len(args) == 2:
            a, b = (args[0].lstrip("?"), args[1].lstrip("?"))
            ta, tb = self.p.type_of.get(a), self.p.type_of.get(b)
            if a not in bound and b not in bound and ta and ta == tb and a != b:
                phrase = pluralize(lexicalize(ta))
                verb = "are not" if neg else "are"
                return f"the two {phrase} {verb} {RECIPROCAL[pred]}"

        table = NEGATIVE if neg else POSITIVE
        if pred not in table:
            self.fallbacks.append(f"unknown predicate {pred}")
            verb = "does not satisfy" if neg else "satisfies"
            frame = "{0} " + verb + " " + pred + (" {1}" if len(args) > 1 else "")
        else:
            frame = table[pred]
        sys_at = SYSTEM_ARG.get(pred)
        nouns = [self.noun(a, bound, bare=(i == sys_at))
                 for i, a in enumerate(args, start=1)]
        text = frame.format(*nouns)
        if nouns and is_plural_subject(nouns[0]):
            # The templates are written singular ("{0} is inside {1}"); an
            # aggregated or plural-shaped subject needs the verb to agree.
            for sg, pl in ((" is ", " are "), (" has ", " have "),
                           (" holds ", " hold "), (" contains ", " contain ")):
                if sg in text:
                    text = text.replace(sg, pl, 1)
                    break
        return text

    # -- ground-atom aggregation ------------------------------------------ #
    def _aggregate(self, leaves: list[tuple[bool, list]], bound: dict[str, str]
                   ) -> list[str]:
        """Collapse sibling atoms that differ only in one instance index.

        ``real(half__log_1) .. real(half__log_8)`` is eight clauses that a person
        says once ("there are eight half logs"). Grouping key: polarity, predicate,
        and the argument tuple with each position folded to its category -- a group
        collapses only if exactly one position actually varies."""
        groups: dict[tuple, list[tuple[bool, list]]] = {}
        order: list[tuple] = []
        for neg, term in leaves:
            key = (neg, term[0], tuple(lexicalize(a) for a in term[1:]))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((neg, term))

        out = []
        for key in order:
            members = groups[key]
            if len(members) == 1:
                neg, term = members[0]
                out.append(self._leaf(term, bound, neg))
                continue
            neg, first = members[0]
            varying = [i for i in range(1, len(first))
                       if len({m[1][i] for m in members}) > 1]
            if len(varying) != 1:
                out.extend(self._leaf(t, bound, n) for n, t in members)
                continue
            i = varying[0]
            phrase = f"{numword(len(members))} {pluralize(lexicalize(first[i]))}"
            merged = list(first)
            merged[i] = "\0count"
            out.append(self._leaf(merged, {**bound, "\0count": phrase}, neg))
        return out

    # -- recursion --------------------------------------------------------- #
    def render(self, term: Term, bound: dict[str, str], neg: bool = False,
               depth: int = 0) -> str:
        if not isinstance(term, list) or not term:
            self.fallbacks.append(f"non-list term {term!r}")
            return str(term)

        head = term[0]

        if head == "not":
            return self.render(term[1], bound, not neg, depth)

        if head in ("and", "or"):
            # `not (and a b)` is `or (not a) (not b)`; pushing the negation inwards
            # is what lets the positive-word negative templates keep working
            # ("closed" instead of "not open").
            conj = (head == "and") != neg
            leaves = [t for t in term[1:] if isinstance(t, list) and t
                      and not any(isinstance(s, list) for s in t)]
            nested = [t for t in term[1:] if t not in leaves]
            parts = self._aggregate([(neg, t) for t in leaves], bound)
            parts += [self.render(t, bound, neg, depth + 1) for t in nested]
            if len(parts) == 1:
                return parts[0]
            # Semicolons only at the top level; nested lists use plain "and"/"or"
            # so the two levels stay visually distinct. A nested disjunction keeps
            # its comma, because "A and B or C and D" does not bracket in English.
            if depth:
                return (" and " if conj else ", or ").join(parts)
            tail = ", and " if conj else ", or "
            return "; ".join(parts[:-1]) + tail + parts[-1]

        if head == "imply":
            a = self.render(term[1], bound, False, depth + 1)
            b = self.render(term[2], bound, neg, depth + 1)
            return f"if {a}, then {b}"

        if head in QUANT:
            return self._quantified(head, term, bound, neg, depth)

        return self._leaf(term, bound, neg)

    def _quantified(self, head: str, term: list, bound: dict[str, str], neg: bool,
                    depth: int) -> str:
        def var_type(spec: list) -> tuple[str, str, int]:
            return spec[0], lexicalize(spec[-1]), self.p.count.get(spec[-1], 0)

        if head in ("forall", "exists"):
            var, noun, n_inst = var_type(term[1])
            # Negating a quantifier swaps it: not(forall x. P) == exists x. not P.
            effective = head if not neg else ("exists" if head == "forall" else "forall")
            if n_inst == 1:
                # Quantifying over a single declared instance: "every plate" and
                # "at least one plate" both just mean "the plate".
                phrase = "the " + noun
            elif effective == "forall":
                phrase = "every " + noun
            else:
                phrase = "at least one " + noun
            return self.render(term[2], {**bound, var: phrase}, neg, depth + 1)

        if head == "forn":
            n = int(term[1][0])
            var, noun, _ = var_type(term[2])
            phrase = f"exactly {numword(n)} {pluralize(noun)}"
            return self.render(term[3], {**bound, var: phrase}, neg, depth + 1)

        if head == "forpairs":
            v1, n1, _ = var_type(term[1])
            v2, n2, _ = var_type(term[2])
            return self.render(term[3],
                               {**bound, v1: "each " + n1, v2: f"a different {n2}"},
                               neg, depth + 1)

        if head == "fornpairs":
            n = int(term[1][0])
            v1, n1, _ = var_type(term[2])
            v2, n2, _ = var_type(term[3])
            return self.render(term[4],
                               {**bound, v1: f"each of {numword(n)} {pluralize(n1)}",
                                v2: f"a different {n2}"}, neg, depth + 1)

        self.fallbacks.append(f"unhandled quantifier {head}")
        return self.render(term[-1], bound, neg, depth)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def render_goal_nl(activity: str, definition_id: int = 0, bddl_dir: str | None = None,
                   room_qualified: bool = True) -> tuple[str, list[str]]:
    """Return ``(sentence, fallbacks)`` for one activity's unground BDDL goal.

    ``fallbacks`` is non-empty when some construct took a generic path; it is
    returned rather than logged so callers (and ``--audit``) can refuse to train
    on prompts that came out mechanical."""
    problem = Problem(activity, definition_id, bddl_dir)
    r = GoalRenderer(problem, room_qualified=room_qualified)
    text = r.render(problem.goal, {})
    return text[:1].upper() + text[1:] + ".", r.fallbacks


def build_nl_prompt(activity: str, goal_nl: str, obs_mode: str = "full") -> str:
    """The user-turn text for the NL-goal condition.

    Deliberately does NOT include the ground atoms, the object list, or the
    ``inroom`` facts: those are what the agent is supposed to establish with
    ``observe``. The activity name stays because a person names the errand."""
    return (
        f"Task: {activity.replace('_', ' ')}\n"
        f"Observability: {obs_mode}\n"
        f"You are done when this is true: {goal_nl}"
    )


# --------------------------------------------------------------------------- #
def _audit(bddl_dir: str | None) -> int:
    """Render every activity; report coverage and every fallback."""
    import collections

    root = bddl_dir or activity_definitions_dir()
    acts = sorted(a for a in os.listdir(root)
                  if os.path.exists(os.path.join(root, a, "problem0.bddl")))
    ok = 0
    reasons: collections.Counter = collections.Counter()
    failed: list[tuple[str, str]] = []
    lengths = []
    for a in acts:
        try:
            text, fb = render_goal_nl(a, bddl_dir=bddl_dir)
        except Exception as ex:                      # noqa: BLE001 — audit reports, not raises
            failed.append((a, f"{type(ex).__name__}: {ex}"))
            continue
        lengths.append(len(text))
        if fb:
            for f in fb:
                reasons[f.split(" ", 2)[0] + " " + f.split(" ")[1]] += 1
        else:
            ok += 1
    print(f"activities: {len(acts)}   clean: {ok}   with fallback: "
          f"{len(acts) - ok - len(failed)}   errors: {len(failed)}")
    print(f"sentence length: mean {sum(lengths) / max(len(lengths), 1):.0f} chars, "
          f"max {max(lengths, default=0)}")
    if reasons:
        print("\nfallbacks:")
        for k, v in reasons.most_common():
            print(f"  {v:5d}  {k}")
    for a, err in failed[:10]:
        print(f"  ERROR {a}: {err}")
    return 0 if not failed else 1


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("activity", nargs="*", help="activity name(s) to render")
    ap.add_argument("--bddl-dir", default=None)
    ap.add_argument("--no-room-qualified", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="render all activities and report fallback coverage")
    ap.add_argument("--dump", metavar="PATH",
                    help="write {activity: sentence} JSON for every activity")
    args = ap.parse_args()

    if args.audit:
        return _audit(args.bddl_dir)

    if args.dump:
        import json

        root = args.bddl_dir or activity_definitions_dir()
        out = {}
        for a in sorted(os.listdir(root)):
            if not os.path.exists(os.path.join(root, a, "problem0.bddl")):
                continue
            try:
                out[a] = render_goal_nl(a, bddl_dir=args.bddl_dir,
                                        room_qualified=not args.no_room_qualified)[0]
            except Exception:                        # noqa: BLE001
                continue
        with open(args.dump, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print(f"wrote {len(out)} sentences -> {args.dump}")
        return 0

    for a in args.activity:
        text, fb = render_goal_nl(a, bddl_dir=args.bddl_dir,
                                  room_qualified=not args.no_room_qualified)
        print(f"# {a}\n{text}")
        if fb:
            print(f"  (fallbacks: {fb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
