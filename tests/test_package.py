"""Package checks: the skill layout, no personal or workspace trace, and a licence row for every model a script loads."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = sorted((ROOT / "skills").iterdir())
TEXT = {".py", ".md", ".json", ".txt", ".toml", ".cfg"}
SKIP_DIRS = {"__pycache__", ".pytest_cache", "refmatch-output"}


def files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not SKIP_DIRS & set(p.parts) and (p.suffix in TEXT or p.name in {"LICENSE", ".gitignore"}):
            yield p


HEADERS = ["Trigger", "Desired outcome", "Procedure", "Stop test", "Failure branches", "Writes", "When NOT to use"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2705\u274C]")


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_skill_follows_the_seven_section_standard(skill):
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    front = re.match(r"---\nname: ([a-z0-9-]+)\ndescription: \"([^\"\n]+)\"\n---\n", text)
    assert front and front.group(1) == skill.name, "frontmatter needs a name and a quoted description"
    headers = re.findall(r"^## (.+)$", text, re.M)
    assert headers == HEADERS
    assert "\u2014" not in text and not EMOJI.search(text)


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_every_path_a_skill_names_exists(skill):
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    for rel in set(re.findall(r"SKILL_DIR/([\w./-]+?)(?=[\s`)\"'<,]|$)", text)):
        rel = rel.rstrip(".")
        assert (skill / rel).exists(), f"{skill.name}/SKILL.md names SKILL_DIR/{rel}, which does not exist"


def test_no_em_dashes_or_emojis_anywhere():
    bad = [str(p.relative_to(ROOT)) for p in files() if "\u2014" in p.read_text(encoding="utf-8") or EMOJI.search(p.read_text(encoding="utf-8"))]
    assert not bad, bad


# Words that would show a personal or workspace link. A hit needs an entry in JUSTIFIED naming the file and why.
LEAKS = re.compile(r"jetahiri|C:\\+Users|gabriel|pharos|teo|workspace\.sqlite|client\.|card_lease|music-app|build #|@gmail|api_key|token|workspace",
                   re.I)
JUSTIFIED = {("LICENSE", 3)}  # the copyright holder is named on purpose (first name and public handle only)


def test_no_personal_or_workspace_trace():
    hits = []
    for p in files():
        if p.name == "test_package.py":
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if LEAKS.search(line) and (str(p.relative_to(ROOT)), n) not in JUSTIFIED:
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


def test_no_script_reaches_into_a_workspace_or_home_folder():
    for p in (ROOT / "skills").rglob("scripts/*.py"):
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"Path\.home\(\)", src) and "import client" not in src, p


def test_every_model_a_script_loads_has_a_licence_row_and_the_excluded_one_is_never_loaded():
    licences = (ROOT / "LICENCES.md").read_text(encoding="utf-8")
    ids = set()
    for p in (ROOT / "skills").rglob("scripts/*.py"):
        src = p.read_text(encoding="utf-8")
        ids |= set(re.findall(r'(?:MODEL|mid)\s*=\s*"([\w.-]+/[\w.-]+)"', src))
        assert "snapshot_download" not in src and "BriaRMBG" not in src and "briaai" not in src, p
    assert ids == {"depth-anything/Depth-Anything-V2-Small-hf", "prs-eth/marigold-normals-v1-1"}, ids
    for i in ids | {"VAST-AI/TripoSG"}:
        assert f"`{i}`" in licences
    assert "RMBG-1.4" in licences.split("## Excluded")[1].split("## Libraries")[0]
    assert "rmbg_net=None" in (ROOT / "skills/photo-to-3d/scripts/image_to_mesh.py").read_text(encoding="utf-8")


def test_every_licence_row_has_a_commercial_answer_and_none_is_non_commercial():
    table = (ROOT / "LICENCES.md").read_text(encoding="utf-8").split("## Excluded")[0].split("## Models the scripts load")[1]
    rows = [r for r in table.splitlines() if r.startswith("| ") and not r.startswith("| Model") and not r.startswith("|---")]
    assert len(rows) == 5
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        assert cells[3].lower().startswith("yes"), r
        assert "2026-10-21" in table and "http" in cells[5], r


def test_bpy_importers_carry_the_gpl_header_and_the_rest_do_not():
    for p in (ROOT / "skills").rglob("scripts/*.py"):
        src = p.read_text(encoding="utf-8")
        imports_bpy = bool(re.search(r"^import bpy|^from bpy", src, re.M))
        assert imports_bpy == ("GPL-3.0-or-later" in src.splitlines()[0]), p
