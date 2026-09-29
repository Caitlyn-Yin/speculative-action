"""Port of the MediaWiki title-normalisation and near-match case ladder.

This module exists so that the frozen local corpus resolves a `search[X]` the
same way `https://en.wikipedia.org/w/index.php?search=X` did in 2019: that URL
runs `SpecialSearch`, which first asks `SearchEngine::getNearMatch()` for a
"Go"-style near match and *redirects to the article* if one is found. Only when
near-match returns nothing does the user see a result list -- which is exactly
the branch `WikiEnv.search_step` detects via `mw-search-result-heading`.

Everything here is a transcription, not an approximation. Line references are
into **MediaWiki REL1_33** (released 2019-06, i.e. contemporaneous with the
2019-08-01 KILT snapshot); the files were read from
`raw.githubusercontent.com/wikimedia/mediawiki/REL1_33/...`:

  includes/search/SearchNearMatcher.php:54-132   the case ladder (below)
  languages/Language.php:2705-2831               ucfirst/uc/lc/ucwords/ucwordbreaks
  includes/title/MediaWikiTitleCodec.php:252-420 splitTitleString (normalisation)
  includes/title/MediaWikiTitleCodec.php:458-476 getTitleInvalidRegex
  includes/Title.php:3110-3116                   Title::capitalize
  includes/DefaultSettings.php:3906              $wgLegalTitleChars

Two configuration facts about the English Wikipedia are baked in, because the
corpus is enwiki-only:
  * `$wgCapitalLinks = true`  -> main-namespace titles are ucfirst'd.
  * `$lang->hasVariants()` is false for `en`, so the `$allSearchTerms` loop in
    SearchNearMatcher.php:58-63,77 has exactly one iteration.
"""

import re

# ---------------------------------------------------------------------------
# Language.php case helpers
#
# The PHP originals branch on Language::isMultibyte() (Language.php:2776),
# which is `strlen($s) !== mb_strlen($s)` -- i.e. "does the string contain any
# non-ASCII byte". `ord()` in PHP is a *byte* value, so the ASCII/multibyte
# split has to be reproduced on the encoded form, not on Python characters.
# ---------------------------------------------------------------------------

_PHP_UCWORDS_DELIMS = " \t\r\n\f\v"

# The PHP byte class `[a-z]|[\xc0-\xff][\x80-\xbf]*` matches one lowercase
# ASCII letter or one whole multibyte UTF-8 character. In Python's str domain
# that is "a lowercase ASCII letter, or any non-ASCII character".
_MB_LOWER = r"(?:[a-z]|[^\x00-\x7f])"


def _is_multibyte(s):
    """Language::isMultibyte -- Language.php:2776."""
    return len(s.encode("utf-8")) != len(s)


def lang_ucfirst(s):
    """Language::ucfirst -- Language.php:2705-2715."""
    if not s:
        return s
    first_byte = s.encode("utf-8")[0]
    if first_byte < 96:          # already uppercase / digit / punctuation
        return s
    # < 128 -> PHP ucfirst (ASCII only); >= 128 -> uc($str, true), i.e.
    # mb_strtoupper of the first character. Both are s[0].upper() here.
    return s[0].upper() + s[1:]


def lang_lcfirst(s):
    """Language::lcfirst -- Language.php:2741-2753."""
    if not s:
        return s
    first_byte = s.encode("utf-8")[0]
    if first_byte >= 128:
        return s[0].lower() + s[1:]
    if first_byte > 96:          # already lowercase
        return s
    return s[0].lower() + s[1:]


def lang_lc(s):
    """Language::lc -- Language.php:2760-2770 (mb_strtolower / strtolower)."""
    return s.lower()


def lang_uc(s):
    """Language::uc -- Language.php:2725-2735 (mb_strtoupper / strtoupper)."""
    return s.upper()


def _php_ucwords(s):
    """PHP's builtin ucwords(): uppercase the first character of each word,
    where a word starts at the string start or after " \\t\\r\\n\\f\\v"."""
    out = []
    cap = True
    for ch in s:
        out.append(ch.upper() if cap else ch)
        cap = ch in _PHP_UCWORDS_DELIMS
    return "".join(out)


def lang_ucwords(s):
    """Language::ucwords -- Language.php:2784-2800."""
    if _is_multibyte(s):
        low = lang_lc(s)
        # /^([a-z]|mb)| ([a-z]|mb)/ with mb_strtoupper($matches[0])
        return re.sub(r"^" + _MB_LOWER + r"| " + _MB_LOWER,
                      lambda m: m.group(0).upper(), low)
    return _php_ucwords(s.lower())


def lang_ucwordbreaks(s):
    """Language::ucwordbreaks -- Language.php:2808-2831."""
    if _is_multibyte(s):
        low = lang_lc(s)
        # $breaks = "[ \-\(\)\}\{\.,\?!]"
        return re.sub(r"^" + _MB_LOWER + r"|[ \-\(\)\}\{\.,?!]" + _MB_LOWER,
                      lambda m: m.group(0).upper(), low)
    # /\b([\w\x80-\xff]+)\b/ with ucfirst(); PCRE without /u, so \w is ASCII.
    return re.sub(r"\b(\w+)\b", lambda m: lang_ucfirst(m.group(1)), s,
                  flags=re.ASCII)


# ---------------------------------------------------------------------------
# MediaWikiTitleCodec::splitTitleString -- normalisation
# ---------------------------------------------------------------------------

# Unicode bidi overrides stripped at MediaWikiTitleCodec.php:268
# (bytes \xE2\x80 followed by \x8E, \x8F or \xAA-\xAE).
_BIDI_RE = re.compile("[‎‏‪-‮]")

# Whitespace collapse, MediaWikiTitleCodec.php:274-278.
_WS_RE = re.compile(
    "[ _  ᠎ -     　]+")

# $wgLegalTitleChars, DefaultSettings.php:3906 --
#   " %!\"$&'()*,\-.\/0-9:;=?@A-Z\\^_`a-z~\x80-\xFF+"
# The \x80-\xFF byte range means "any non-ASCII character" once decoded, so the
# illegal set is the ASCII characters *outside* the class above:
#   control chars, and  # < > [ ] { } |
_ILLEGAL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f#<>\[\]{}|]")

# The other three alternatives of getTitleInvalidRegex (MediaWikiTitleCodec.php:467-471)
_ILLEGAL_SEQ_RE = re.compile(
    r"%[0-9A-Fa-f]{2}"
    "|&[A-Za-z0-9\u0080-\U0010ffff]+;"
    r"|&#[0-9]+;"
    r"|&#x[0-9A-Fa-f]+;")

_PREFIX_RE = re.compile(r"^(.+?)_*:_*(.*)$", re.S)

# enwiki namespace names and aliases as of 2019 (canonical + local aliases).
# A search term whose colon-prefix hits one of these is NOT a main-namespace
# title, and the KILT corpus contains main-namespace articles only, so we
# report "no local title" -- see LocalWiki._resolve for how that is surfaced.
_NAMESPACE_PREFIXES = {
    "media", "special", "talk", "user", "user talk", "project", "project talk",
    "wikipedia", "wikipedia talk", "wp", "wt", "file", "file talk",
    "image", "image talk", "mediawiki", "mediawiki talk", "template",
    "template talk", "help", "help talk", "category", "category talk",
    "portal", "portal talk", "book", "book talk", "draft", "draft talk",
    "education program", "education program talk", "timedtext",
    "timedtext talk", "module", "module talk", "gadget", "gadget talk",
    "gadget definition", "gadget definition talk", "topic",
}

# Interwiki prefixes are matched loosely: a colon-prefix that is a known
# interwiki makes Title::isExternal() true, and SearchNearMatcher.php:89
# returns it as a match *without* consulting the local page table. Such a
# search never renders a result list, so it is not a "miss" either. Only the
# handful an agent could plausibly emit are listed.
_INTERWIKI_PREFIXES = {
    "en", "de", "fr", "es", "it", "ja", "zh", "ru", "pt", "nl", "pl", "sv",
    "wikt", "wiktionary", "commons", "w", "wikipedia:en", "meta", "s",
    "wikisource", "q", "wikiquote", "b", "wikibooks", "n", "wikinews",
    "v", "wikiversity", "d", "wikidata", "species", "mw", "phab", "arxiv",
    "doi", "google", "imdbtitle",
}


class TitleError(ValueError):
    """MalformedTitleException equivalent."""


#: returned by :func:`mw_normalize_title` when the term resolves to a title
#: that cannot be a main-namespace enwiki article (other namespace, interwiki).
NOT_MAIN_NS = object()


def mw_normalize_title(text):
    """Port of ``MediaWikiTitleCodec::splitTitleString($text, NS_MAIN)``
    (MediaWikiTitleCodec.php:252-420) specialised to enwiki main namespace.

    Returns the canonical page title with spaces (the form KILT stores), or
    raises :class:`TitleError` for a malformed title, or returns
    :data:`NOT_MAIN_NS` when a namespace / interwiki prefix takes the title out
    of the article namespace.
    """
    if not isinstance(text, str):
        raise TitleError("not a string")

    dbkey = text.replace(" ", "_")                       # :253
    dbkey = _BIDI_RE.sub("", dbkey)                      # :268
    dbkey = _WS_RE.sub("_", dbkey)                       # :274
    dbkey = dbkey.strip("_")                             # :279

    if dbkey == "":
        raise TitleError("title-invalid-empty")          # :296

    if dbkey[0] == ":":                                  # :290-294
        dbkey = dbkey[1:].strip("_")
        if dbkey == "":
            raise TitleError("title-invalid-empty")
    else:
        m = _PREFIX_RE.match(dbkey)                      # :301-363
        if m:
            prefix = m.group(1).replace("_", " ").lower()
            if prefix in _NAMESPACE_PREFIXES or prefix in _INTERWIKI_PREFIXES:
                return NOT_MAIN_NS
            # Unrecognised prefix -> the colon stays part of the title.

    frag = dbkey.find("#")                               # :365-372
    if frag != -1:
        dbkey = re.sub(r"_*$", "", dbkey[:frag])
        if dbkey == "":
            raise TitleError("title-invalid-empty")

    if _ILLEGAL_CHAR_RE.search(dbkey) or _ILLEGAL_SEQ_RE.search(dbkey):
        raise TitleError("title-invalid-characters")     # :374-379

    if "." in dbkey and (                                # :384-397
        dbkey in (".", "..")
        or dbkey.startswith("./") or dbkey.startswith("../")
        or "/./" in dbkey or "/../" in dbkey
        or dbkey.endswith("/.") or dbkey.endswith("/..")
    ):
        raise TitleError("title-invalid-relative")

    if "~~~" in dbkey:                                   # :399-402
        raise TitleError("title-invalid-magic-tilde")

    if len(dbkey.encode("utf-8")) > 255:                 # :404-412
        raise TitleError("title-invalid-too-long")

    dbkey = lang_ucfirst(dbkey)                          # :419 via Title::capitalize
    return dbkey.replace("_", " ")


def near_match_variants(searchterm):
    """The ordered candidate titles tried by ``SearchNearMatcher::getNearMatchInternal``
    (SearchNearMatcher.php:54-132), for ``en`` (no language variants).

    Yields ``(label, title)`` pairs; the caller checks each against the page
    table and takes the first that exists. This is *only* the part of the PHP
    method that consults the local page table -- the Media/Special/User/File/
    MediaWiki-namespace fallbacks after the loop (SearchNearMatcher.php:134-169)
    cannot hit a main-namespace article and are handled in
    :func:`mw_normalize_title` / the caller.

    Order and labels:
      exact        Title::newFromText($term)              :79
      lc           Title::newFromText($lang->lc($term))    :104
      ucwords      Title::newFromText($lang->ucwords(...)) :110
      uc           Title::newFromText($lang->uc($term))    :116
      ucwordbreaks Title::newFromText($lang->ucwordbreaks) :122
    """
    # SearchNearMatcher.php:73 -- '' and anything starting with '#' is a miss.
    if searchterm == "" or searchterm[0] == "#":
        return []

    out = []
    seen = set()
    # :79-82 -- a malformed *exact* term aborts the whole ladder (`return null`).
    try:
        exact = mw_normalize_title(searchterm)
    except TitleError:
        return []
    if exact is NOT_MAIN_NS:
        return [("exact", NOT_MAIN_NS)]
    out.append(("exact", exact))
    seen.add(exact)

    # :104-125 -- the four case variants. Unlike the exact step, a malformed
    # variant is simply skipped (`if ( $title && $title->exists() )`).
    for label, fn in (("lc", lang_lc),
                      ("ucwords", lang_ucwords),
                      ("uc", lang_uc),
                      ("ucwordbreaks", lang_ucwordbreaks)):
        try:
            cand = mw_normalize_title(fn(searchterm))
        except TitleError:
            continue
        if cand is NOT_MAIN_NS or cand in seen:
            continue
        seen.add(cand)
        out.append((label, cand))

    # :165-169 -- after every variant has failed, a fully quoted term is
    # retried without its quotes. Appending its candidates at the end of the
    # ordered list reproduces "only if nothing above matched".
    m = re.match(r'^"([^"]+)"$', searchterm)
    if m:
        for label, cand in near_match_variants(m.group(1)):
            if cand is NOT_MAIN_NS or cand in seen:
                continue
            seen.add(cand)
            out.append(("dequote/" + label, cand))
    return out
