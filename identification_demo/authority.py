"""Editable, evidence-based normalization for issuing-authority text."""
import json
import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


logger = logging.getLogger(__name__)

LEADER_TITLE = re.compile(
    r"^(?:(?:k\s*\.?\s*t\.?|t\s*\.?\s*l\.?|q\.?|pho|quyen|thua\s*lenh)\s*[.:/-]*\s*)*"
    r"(?:cuc\s*truong|giam\s*doc|bo\s*truong|truong\s*phong)\b[\s:.,/-]*"
)


def strip_authority_title(value):
    """The issuer is the institution, rather than the title of its signatory."""
    value = re.sub(r"\s+", " ", value).strip()
    for _ in range(4):
        folded = "".join(char for char in unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
                         if unicodedata.category(char) != "Mn")
        match = LEADER_TITLE.match(folded)
        if not match:
            break
        value = value[match.end():].strip()
    return value


def has_institution_name(value):
    """An administrative-description fragment alone cannot identify its issuer."""
    value = compact(strip_authority_title(value))
    return bool(re.match(r"^(?:cuc|cong\s*an|bo|so|phong|uy\s*ban|"
                         r"ministry|department|police\s*department)\b", value)
                and not re.search(r"\b(?:tinh|thanh\s*pho)$", value))


def compact(text):
    text = text.lower().replace("đ", "d")
    text = "".join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


@lru_cache(maxsize=8)
def _read_rules(path, _mtime, _size, _inode):
    rules = json.loads(Path(path).read_text())
    if not isinstance(rules, dict):
        raise ValueError("Rules must be an object")
    threshold = float(rules.get("min_similarity", 0.9))
    if not 0.85 <= threshold <= 1:
        raise ValueError("min_similarity must be between 0.85 and 1")
    candidates = rules.get("authorities", [])
    if not isinstance(candidates, list):
        raise ValueError("authorities must be a list")
    for candidate in candidates:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str) or not candidate["name"].strip()
                or not isinstance(candidate.get("required_fragments"), list)
                or len(candidate["required_fragments"]) < 2
                or not all(isinstance(value, str) and compact(value) for value in candidate["required_fragments"])):
            raise ValueError("Each authority needs a name and at least two required fragments")
    return rules


def authority_rules():
    path = Path(os.getenv("ISSUING_AUTHORITY_RULES_FILE", str(Path(__file__).with_name("config") / "issuing_authority_rules.json")))
    try:
        stat = path.stat()
        return _read_rules(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except (OSError, ValueError, TypeError):
        logger.warning("Issuing authority rules unavailable or invalid; preserving recognized text")
        return {"enabled": False, "authorities": []}


def normalize_authority(value):
    value = strip_authority_title(value)
    rules = authority_rules()
    if not value or not rules.get("enabled", True):
        return value
    observed = compact(value)
    matches = []
    for candidate in rules["authorities"]:
        name = compact(strip_authority_title(candidate["name"]))
        fragments = [cleaned for fragment in candidate["required_fragments"] if (cleaned := strip_authority_title(fragment))]
        if not fragments:
            continue
        if all(re.search(r"\b" + re.escape(compact(fragment)) + r"\b", observed)
               for fragment in fragments):
            ratio = SequenceMatcher(None, observed, name).ratio()
            matches.append((ratio, strip_authority_title(candidate["name"])))
    matches.sort(reverse=True)
    if (matches and matches[0][0] >= rules.get("min_similarity", 0.9)
            and (len(matches) == 1 or matches[0][0] - matches[1][0] >= 0.04)):
        return matches[0][1]
    return value


def incomplete_authority(value):
    """A recognized institution head still needs its printed continuation."""
    if not has_institution_name(value):
        return True
    observed = compact(strip_authority_title(value))
    if observed in {"cuc", "cuc canh sat", "cong an", "bo", "so", "phong"}:
        return True
    if re.search(r"\b(?:ve|va|tinh|thanh pho)$", observed):
        return True
    # A known name is evidence that a prefix is incomplete, never evidence to
    # fill its missing words. Recovery must read those words from image pixels.
    for candidate in authority_rules().get("authorities", []):
        name = compact(strip_authority_title(candidate["name"]))
        if name.startswith(observed + " "):
            return True
        # Missing middle lines are also incomplete: "CỤC CẢNH SÁT VỀ TRẬT
        # TỰ XÃ HỘI" must not become complete just because it has an ending.
        if observed != name and observed.split()[:3] == name.split()[:3]:
            remaining = iter(name.split())
            if all(any(word == token for word in remaining) for token in observed.split()):
                return True
    return False


def authority_word_score(value):
    vocabulary = set("cuc truong canh sat quan ly hanh chinh ve trat tu xa hoi dang ky cu tru va du lieu quoc gia dan cong an giam doc tinh thanh pho bo ministry public security police department administrative social order".split())
    for candidate in authority_rules().get("authorities", []):
        vocabulary.update(compact(candidate["name"]).split())
    return len(set(compact(value).split()) & vocabulary)
