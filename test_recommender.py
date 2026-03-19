"""
Tests voor de WasteLess Aanbeveler
===================================
Runt vier controles op het systeem zonder dat je de data hoeft te inspecteren.

Gebruik:
    python test_recommender.py
"""

from recommender import (
    WasteLessRecommender, build_pantry, fuzzy_lookup,
    urgency_weight, vectorized_coverage, match_pantry_to_ingredients, normalize
)
import pandas as pd
from datetime import datetime, timedelta

USERS_TO_TEST = [1, 21, 61]
TOP_N = 5

PASS = "✓ PASS"
FAIL = "✗ FAIL"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


# ── Test 1: Scores altijd tussen 0 en 1 ──────────────────────────────────────

def test_scores_in_range(rec):
    print("\n[1] Scores tussen 0 en 1")
    ok = True
    for uid in USERS_TO_TEST:
        result = rec.recommend(uid, top_n=TOP_N)
        for lijst, df in result.items():
            if df.empty:
                continue
            boven = df[df["score_final"] > 1.0]
            onder = df[df["score_final"] < 0.0]
            ok &= check(
                f"user {uid} / {lijst}: score_final in [0, 1]",
                boven.empty and onder.empty,
                f"{len(boven)} scores > 1, {len(onder)} scores < 0"
            )
            # rating-score mag iets hoger door de boost (max +20%)
            boven_r = df[df["score_with_rating"] > 1.25]
            ok &= check(
                f"user {uid} / {lijst}: score_with_rating redelijk (<1.25)",
                boven_r.empty,
                f"{len(boven_r)} scores > 1.25"
            )
    return ok


# ── Test 2: Geen foute fuzzy matches ─────────────────────────────────────────

def test_no_bad_fuzzy_matches(rec):
    """
    Controleer een lijst bekende probleemgevallen die vóór de fix
    verkeerd matchten (witte bonen → witte wijn, zwarte olijven → zwarte peper).
    """
    print("\n[2] Geen foute fuzzy matches")
    ok = True

    # Pantry-sleutels moeten genormaliseerd zijn (zoals build_pantry ze aanmaakt)
    # "witte wijn" → normalize() → "wijn", "zwarte peper" → "peper"
    from recommender import normalize
    pantry = {
        normalize("witte wijn"):   {"quantity": 500, "urgency": 1.0},
        normalize("zwarte peper"): {"quantity": 10,  "urgency": 1.0},
        normalize("rijst"):        {"quantity": 500, "urgency": 1.0},
    }
    # Controleer wat de genormaliseerde sleutels zijn
    print(f"    (pantry sleutels: {list(pantry.keys())})")

    foute_matches = [
        ("witte bonen",         "witte wijn"),
        ("witte chocolade",     "witte wijn"),
        ("witte sesamzaadjes",  "witte wijn"),
        ("zwarte olijven",      "zwarte peper"),
        ("zwarte bonen",        "zwarte peper"),
        ("zwarte sesamzaadjes", "zwarte peper"),
        ("witte rijst",         "witte wijn"),
    ]

    for ingredient, verkeerde_match in foute_matches:
        # normalize() zoals de recommender dat ook doet
        qty, _ = fuzzy_lookup(normalize(ingredient), pantry)
        # als qty > 0 is er een match gevonden — dat is wat we willen voorkomen
        # voor deze specifieke combinaties
        matched = qty > 0
        # "witte rijst" mag matchen op rijst (dat is correct), maar niet op witte wijn
        if ingredient == "witte rijst":
            # aparte pantry met rijst maar zonder witte wijn
            pantry2 = {normalize("rijst"): {"quantity": 500, "urgency": 1.0}}
            qty2, _ = fuzzy_lookup(normalize(ingredient), pantry2)
            ok &= check(
                f"'{ingredient}' matcht correct op 'rijst'",
                qty2 > 0,
                "verwacht match op rijst"
            )
        else:
            ok &= check(
                f"'{ingredient}' matcht NIET op '{verkeerde_match}'",
                not matched,
                f"toch gematcht (qty={qty})"
            )
    return ok


# ── Test 3: Urgente items komen hoger ─────────────────────────────────────────

def test_urgent_items_rank_higher(rec):
    """
    Maak twee identieke voorraden, één met een ingredient dat morgen verloopt.
    Het recept dat dat ingredient gebruikt moet hoger scoren.
    """
    print("\n[3] Urgente items leiden tot hogere score")
    ok = True

    pantry_normaal = {
        "kipfilet": {"quantity": 300, "urgency": 1.0},
        "citroen":  {"quantity": 100, "urgency": 1.0},
    }
    pantry_urgent = {
        "kipfilet": {"quantity": 300, "urgency": 2.0},   # verloopt morgen
        "citroen":  {"quantity": 100, "urgency": 1.0},
    }

    recept = ["300 g kipfilet", "1 stuks citroen", "2 el olijfolie",
              "1 tl zout", "1 tl peper", "1 teen knoflook"]

    from recommender import parse_ingredient
    parsed = [parse_ingredient(l) for l in recept]

    # Bouw lookup dicts {naam: (quantity, urgency)} voor vectorized_coverage
    lookup_normaal = {name: (pantry_normaal[name]["quantity"], pantry_normaal[name]["urgency"])
                      if name in pantry_normaal else (0.0, 1.0) for name, _ in parsed}
    lookup_urgent  = {name: (pantry_urgent[name]["quantity"], pantry_urgent[name]["urgency"])
                      if name in pantry_urgent else (0.0, 1.0) for name, _ in parsed}

    scores_n, *_ = vectorized_coverage([parsed], lookup_normaal)
    scores_u, *_ = vectorized_coverage([parsed], lookup_urgent)
    score_normaal = float(scores_n[0])
    score_urgent  = float(scores_u[0])

    ok &= check(
        "urgent pantry scoort hoger dan normaal pantry",
        score_urgent > score_normaal,
        f"urgent={score_urgent:.3f}, normaal={score_normaal:.3f}"
    )

    # urgency_weight geeft hogere waarde naarmate vervaldatum dichterbij is
    gisteren = datetime.today() - timedelta(days=1)
    morgen   = datetime.today() + timedelta(days=1)
    volgende_maand = datetime.today() + timedelta(days=40)

    ok &= check("verlopen item krijgt hoogste urgency (2.2)",
                urgency_weight(gisteren) == 2.2)
    ok &= check("morgen-item krijgt urgency 2.0",
                urgency_weight(morgen) == 2.0)
    ok &= check("ver-weg item krijgt urgency 1.0",
                urgency_weight(volgende_maand) == 1.0)

    return ok


# ── Test 4: Bijna-klaar filter werkt ─────────────────────────────────────────

def test_almost_done_filter(rec):
    print("\n[4] Bijna-klaar filter")
    ok = True
    for uid in USERS_TO_TEST:
        result = rec.recommend(uid, top_n=TOP_N, only_almost_done=True)
        df = result["match"]
        if df.empty:
            print(f"  ~  user {uid}: geen bijna-klaar recepten gevonden (kan kloppen)")
            continue
        niet_bijna = df[df["almost_done"] == False]
        ok &= check(
            f"user {uid}: alle resultaten zijn 'bijna klaar'",
            niet_bijna.empty,
            f"{len(niet_bijna)} recepten zonder bijna-klaar vlag"
        )
        # bijna-klaar betekent >= 60% volledig gedekt
        te_laag = df[df["full"] / df["total"].clip(lower=1) < 0.60]
        ok &= check(
            f"user {uid}: dekking >= 60% voor alle bijna-klaar recepten",
            te_laag.empty,
            f"{len(te_laag)} recepten onder 60% dekking"
        )
    return ok




# ── Test 5: Kleur-specifieke matches ─────────────────────────────────────────

def test_colour_matches(rec):
    """
    Controleer dat kleurgevoelige ingrediënten correct wel/niet matchen.
    Rode wijn mag NIET matchen op witte wijn.
    Rode wijn MAG matchen op rode wijn (triviale match).
    Rode wijnazijn is géén rode wijn.
    """
    from recommender import normalize
    print("\n[5] Kleur-specifieke matches")
    ok = True

    # (recept-ingredient, pantry-item, verwacht_match: True/False, reden)
    cases = [
        ("rode wijn",       "rode wijn",       True,  "zelfde ingredient"),
        ("witte wijn",      "witte wijn",       True,  "zelfde ingredient"),
        ("rode wijn",       "witte wijn",       False, "andere kleur wijn"),
        ("witte wijn",      "rode wijn",        False, "andere kleur wijn"),
        ("rode wijn",       "rode wijnazijn",   False, "wijn ≠ wijnazijn"),
        ("witte wijn",      "witte wijnazijn",  False, "wijn ≠ wijnazijn"),
        ("rode paprika",    "rode wijn",        False, "paprika ≠ wijn"),
        ("rode uien",       "rode wijn",        False, "uien ≠ wijn"),
        ("witte bonen",     "witte wijn",       False, "bonen ≠ wijn"),
        ("witte chocolade", "witte wijn",       False, "chocolade ≠ wijn"),
        ("rode kool",       "rode wijn",        False, "kool ≠ wijn"),
        # AH-specifieke productnamen uit de echte dataset
        ("ah rode wijn",              "rode wijn",       True,  "AH-merk prefix"),
        ("ah witte wijn",             "witte wijn",      True,  "AH-merk prefix"),
        ("droge witte wijn",          "witte wijn",      True,  "droge = stopwoord"),
        ("italiaanse rode wijn",      "rode wijn",       True,  "herkomst prefix"),
        ("soepele rode wijn",         "rode wijn",       True,  "bijvoeglijk naamwoord"),
        ("spaanse witte wijn",        "witte wijn",      True,  "herkomst prefix"),
        ("vegan rode wijn",           "rode wijn",       True,  "vegan prefix"),
        # Moet NIET matchen
        ("rode wijn",                 "witte wijn",      False, "andere kleur"),
        ("rode wijnazijn",            "rode wijn",       False, "azijn ≠ wijn"),
    ]

    for ingredient, pantry_item, verwacht, reden in cases:
        pantry = {normalize(pantry_item): {"quantity": 500, "urgency": 1.0}}
        qty, _ = fuzzy_lookup(normalize(ingredient), pantry)
        gematcht = qty > 0
        label = "matcht" if verwacht else "matcht NIET"
        ok &= check(
            f"'{ingredient}' {label} op '{pantry_item}'  ({reden})",
            gematcht == verwacht,
            f"verwacht={'match' if verwacht else 'geen match'}, gekregen={'match' if gematcht else 'geen match'}"
        )
    return ok

# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("WasteLess Recommender — test suite")
    print("=" * 45)

    rec = WasteLessRecommender()

    resultaten = [
        test_scores_in_range(rec),
        test_no_bad_fuzzy_matches(rec),
        test_urgent_items_rank_higher(rec),
        test_almost_done_filter(rec),
        test_colour_matches(rec),
    ]

    geslaagd = sum(resultaten)
    print(f"\n{'='*45}")
    print(f"  {geslaagd}/{len(resultaten)} tests geslaagd")
    if geslaagd < len(resultaten):
        print("  Zie ✗ FAIL regels hierboven voor details.")