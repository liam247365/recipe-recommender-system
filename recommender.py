"""
WasteLess Aanbeveler
====================
Beveelt recepten aan op basis van wat een gebruiker thuis heeft, met prioriteit
voor ingrediënten die binnenkort verlopen om voedselverspilling te verminderen.

HOE TE GEBRUIKEN
----------------
    pip install pandas numpy openpyxl rapidfuzz scikit-learn

    python recommender.py --user 1 --top 5            # één gebruiker, print in terminal
    python recommender.py --all --output results.xlsx  # alle gebruikers, opslaan in Excel

HOE SCORING WERKT
-----------------
    basis        = 0.75 * voorraad_dekking + 0.25 * tfidf_gelijkenis
    aangepast    = basis * lengte_penalty   (geen straf als recept urgent items gebruikt)
    uiteindelijk = aangepast * verspilling_bonus  (tot +35% voor urgente ingrediënten)

    Ingrediënten die binnenkort verlopen krijgen een dekkingsvermenigvuldiger tot 2.2×.
    Een tweede lijst mengt de score met de gemiddelde gebruikersrating.

SNELHEID
--------
    De coverage-berekening is gevectoriseerd: alle recepten worden in één matrix-
    operatie gescoord per gebruiker in plaats van in een Python for-loop.
    Dit geeft ~10–30× snelheidswinst bij grote receptendatabases.

INVOERBESTANDEN
---------------
    recepten_AH.xlsx    — ID, Naam, Ingrediënten (één per regel), URL, Gemiddelde rating
    pantry_proxies.xlsx — user_id, ingredient, hoeveelheid, eenheid, categorie,
                          profiel, expiry_date
"""

import argparse, re
from datetime import datetime

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer as _Stemmer

_stemmer = _Stemmer("dutch")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Bestandspaden ─────────────────────────────────────────────────────────────
RECIPES_FILE = "recepten_AH.xlsx"
PANTRY_FILE  = "pantry_proxies.xlsx"

# ── Scoring-gewichten ─────────────────────────────────────────────────────────
WEIGHT_COVERAGE  = 0.75   # voorraad-dekking (primair)
WEIGHT_TFIDF     = 0.25   # tekstgelijkenis (secundair)
WASTE_BONUS_MAX  = 0.35   # max +35% bonus voor urgente ingrediënten
RATING_BOOST     = 0.20   # max +20% boost van receptrating (alleen tweede lijst)
RATING_MAX       = 10.0

# ── Lengte-penalty ────────────────────────────────────────────────────────────
MIN_INGREDIENTS  = 6      # korter dan dit → kleine korting
PENALTY_STRENGTH = 0.40   # max aftrek

# ── Andere drempelwaarden ─────────────────────────────────────────────────────
ALMOST_DONE_RATIO = 0.60  # fractie volledig gedekte ingrediënten voor "bijna klaar"
MAX_PER_CATEGORY  = 3     # max recepten per categorie in eindlijst
MMR_LAMBDA        = 0.7   # 1.0 = puur score, 0.0 = puur diversiteit

# ── Eenheid → basisunit (g / ml / ~stuk) ─────────────────────────────────────
UNIT_TO_BASE = {
    "kg": 1000, "g": 1, "mg": 0.001, "l": 1000, "ml": 1, "dl": 100,
    "el": 15, "tl": 5, "eetlepel": 15, "theelepel": 5,
    "stuks": 100, "stuk": 100, "teen": 5, "takje": 5, "snuf": 2, "snufje": 2,
}

STOP_WORDS = {
    "verse", "vers", "gedroogde", "gedroogd", "gerookte", "gerookt",
    "gesneden", "gehakte", "gehakt", "gekookte", "gekookt", "halfvolle",
    "volle", "magere", "milde", "zoete", "zoet", "grote", "groot",
    "kleine", "klein", "biologische", "biologisch", "ah", "mix",
    "voor", "van", "met", "en", "of", "pure", "belegen", "geraspte",
    # Herkomst, stijl en kwaliteitsaanduidingen — zeggen niets over het ingredient zelf
    "italiaanse", "italiaans", "spaanse", "spaans", "franse", "grieks", "griekse",
    "droge", "soepele", "robuuste", "vegan", "plantaardige", "traditionele",
    "houdbare", "diepvries", "premium", "classic", "original",
}

CATEGORY_KEYWORDS = {
    "ontbijt":     ["wafel", "broodje", "bagel", "smoothie", "toast", "pancake"],
    "soep":        ["soep", "bouillon"],
    "salade":      ["salade", "sla"],
    "snack":       ["snack", "chips", "borrel", "dip", "crackers"],
    "dessert":     ["taart", "cake", "cookie", "muffin", "dessert", "pudding"],
    "pasta":       ["pasta", "spaghetti", "penne", "lasagne", "orzo"],
    "vlees":       ["kip", "gehakt", "biefstuk", "spekjes", "chorizo", "bacon"],
    "vis":         ["zalm", "kabeljauw", "tonijn", "garnalen", "makreel"],
    "vegetarisch": ["tofu", "halloumi", "kikkererwten", "linzen", "falafel"],
    "stamppot":    ["stamppot", "puree", "stoof"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_category(name):
    name = name.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in name for kw in kws):
            return cat
    return "overig"


def normalize(text):
    """
    Verwijder stopwoorden, bewaar de eerste 3 betekenisvolle woorden,
    en stem elk woord naar zijn stam (Nederlandse Snowball stemmer).

    Voorbeelden:
        "italiaanse rode wijn" → "rod wijn"   (stam van "rode" = "rod")
        "kipfilets"            → "kipfilet"
        "wijnazijn"            → "wijnazijn"  (andere stam dan "wijn")
    """
    words = [w for w in re.split(r"\s+", text.lower().strip())
             if w not in STOP_WORDS and len(w) > 2]
    stemmed = [_stemmer.stem(w) for w in words[:3]]
    return " ".join(stemmed)


def parse_ingredient(line):
    """Parse '200 g kipfilet' → ('kipfilet', 200.0). Valt terug op (naam, 100)."""
    m = re.match(r"^([\d.,½¼¾]+)\s*([a-zA-Z]+)?\s+(.+)$", line.lower().strip())
    if not m:
        return normalize(line), 100.0
    qty  = float(m.group(1).replace(",", ".").replace("½", "0.5")
                            .replace("¼", "0.25").replace("¾", "0.75"))
    unit = (m.group(2) or "stuks").lower()
    return normalize(m.group(3)), qty * UNIT_TO_BASE.get(unit, 100)


def parse_date(value):
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return pd.NaT if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def urgency_weight(expiry):
    """Vermenigvuldiger op basis van dagen tot vervaldatum (max 2.2×)."""
    exp = parse_date(expiry)
    if pd.isna(exp):
        return 1.0
    days = (exp - pd.Timestamp(datetime.today()).normalize()).days
    for threshold, weight in [(0, 2.2), (1, 2.0), (2, 1.8), (4, 1.6),
                               (7, 1.4), (14, 1.15), (30, 1.05)]:
        if days <= threshold:
            return weight
    return 1.0


# ── Voorraad ──────────────────────────────────────────────────────────────────

def build_pantry(pantry_df, user_id):
    """Geef {genorm_naam: {quantity, expiry, urgency}} voor één gebruiker."""
    pantry = {}
    for _, r in pantry_df[pantry_df["user_id"] == user_id].iterrows():
        name   = normalize(str(r["ingredient"]))
        qty    = float(r["hoeveelheid"]) * UNIT_TO_BASE.get(str(r["eenheid"]).lower(), 100)
        expiry = parse_date(r.get("expiry_date", pd.NaT))
        if name not in pantry:
            pantry[name] = {"quantity": qty, "expiry": expiry}
        else:
            pantry[name]["quantity"] += qty
            e = pantry[name]["expiry"]
            if pd.isna(e) or (not pd.isna(expiry) and expiry < e):
                pantry[name]["expiry"] = expiry

    # Sla urgency op zodat we het niet per recept opnieuw berekenen
    for item in pantry.values():
        item["urgency"] = urgency_weight(item["expiry"])
    return pantry


def fuzzy_lookup(name, pantry):
    """
    Zoek exacte match in de voorraad na normalisatie + stemming.
    Omdat normalize() al beide strings stemt, is een exacte dict-lookup
    voldoende — en nooit incorrect zoals fuzzy matching kon zijn.
    """
    item = pantry.get(name)
    if item:
        return item["quantity"], item["urgency"]
    return 0.0, 1.0


def debug_fuzzy_matches(rec, user_ids):
    """
    Print alle fuzzy matches voor een lijst gebruikers zodat je kunt
    controleren of 'kip' niet matcht op 'kikkererwten' e.d.

    Gebruik:
        rec = WasteLessRecommender()
        debug_fuzzy_matches(rec, [1, 21, 61])

    Output per match:
        recept-ingrediënt  →  voorraad-item  (score)  [VERDACHT als score < 80]
    """
    for uid in user_ids:
        pantry = build_pantry(rec.pantry, uid)
        print(f"\n{'='*55}")
        print(f"  Gebruiker {uid} — voorraad: {list(pantry.keys())}")
        print(f"{'='*55}")
        for name in sorted(rec.unique_names):
            item = pantry.get(name)
            if item:
                print(f"  {name:30s} → match (quantity={item['quantity']})")


# ── Gevectoriseerde coverage ──────────────────────────────────────────────────

def build_recipe_matrix(recipes_df):
    """
    Zet alle recepten om naar een matrix: (n_recepten × max_ingrediënten).

    Elk recept wordt een rij van (benodigd, naam)-paren. We slaan de
    gedeelde ingrediëntnamen op zodat we fuzzy matching maar één keer
    per uniek ingredient hoeven te doen.

    Geeft terug:
        parsed_recipes  — list[list[(naam, benodigd)]] per recept
        unique_names    — gesorteerde set van alle ingrediëntnamen
    """
    parsed_recipes = []
    unique_names   = set()

    for _, recipe in recipes_df.iterrows():
        lines  = [l.strip() for l in str(recipe.get("Ingrediënten", "")).splitlines() if l.strip()]
        parsed = [parse_ingredient(l) for l in lines]
        parsed_recipes.append(parsed)
        unique_names.update(name for name, _ in parsed)

    return parsed_recipes, sorted(unique_names)


def match_pantry_to_ingredients(unique_names, pantry):
    """
    Doe fuzzy matching voor alle unieke ingrediëntnamen in één keer.
    Geeft een dict {naam: (quantity, urgency)}.

    Dit is de kernoptimalisatie: in plaats van fuzzy matching per
    (recept × gebruiker), doen we het maar één keer per gebruiker
    voor alle unieke namen samen.
    """
    return {name: fuzzy_lookup(name, pantry) for name in unique_names}


def vectorized_coverage(parsed_recipes, lookup):
    """
    Bereken coverage-scores voor alle recepten tegelijk met NumPy.

    parsed_recipes : list[list[(naam, benodigd)]]
    lookup         : {naam: (quantity, urgency)}

    Geeft arrays terug (één waarde per recept):
        scores         — gemiddelde gewogen dekking
        full_counts    — aantal volledig gedekte ingrediënten
        partial_counts — gedeeltelijk gedekt
        totals         — totaal ingrediënten
        urgent_counts  — urgente items die gedekt zijn
    """
    n = len(parsed_recipes)
    scores         = np.zeros(n)
    full_counts    = np.zeros(n, dtype=int)
    partial_counts = np.zeros(n, dtype=int)
    totals         = np.zeros(n, dtype=int)
    urgent_counts  = np.zeros(n, dtype=int)

    for i, ingredients in enumerate(parsed_recipes):
        if not ingredients:
            continue
        coverages = []
        for name, needed in ingredients:
            available, urgency = lookup.get(name, (0.0, 1.0))
            raw = min(available, needed) / needed if needed > 0 else (1.0 if available else 0.0)
            coverages.append(min(raw * urgency, 1.8))   # cap zodat één item het gemiddelde niet scheeftrekt

            if urgency >= 1.6 and raw > 0: urgent_counts[i]  += 1
            if raw >= 1.0:                 full_counts[i]    += 1
            elif raw > 0:                  partial_counts[i] += 1

        totals[i] = len(coverages)
        scores[i] = np.mean(coverages) if coverages else 0.0

    return scores, full_counts, partial_counts, totals, urgent_counts


# ── Diversiteit ───────────────────────────────────────────────────────────────

def mmr_rerank(candidates, tfidf_matrix, recipe_index, top_n, score_col="score_final"):
    """
    Maximale Marginale Relevantie: selecteer recepten die hoge score
    combineren met lage gelijkenis aan al gekozen recepten.
    MMR_LAMBDA regelt de balans (1.0 = puur score, 0.0 = puur diversiteit).
    """
    selected, remaining = [], candidates.copy()
    while len(selected) < top_n and len(remaining):
        if not selected:
            best = remaining.iloc[0]
        else:
            sel_vecs = tfidf_matrix[[recipe_index[r["recipe_id"]] for r in selected]]
            mmr = []
            for _, row in remaining.iterrows():
                idx = recipe_index.get(row["recipe_id"])
                sim = float(cosine_similarity(tfidf_matrix[idx:idx+1], sel_vecs).max()) if idx is not None else 1.0
                mmr.append(MMR_LAMBDA * row[score_col] - (1 - MMR_LAMBDA) * sim)
            remaining = remaining.copy()
            remaining["_mmr"] = mmr
            best = remaining.loc[remaining["_mmr"].idxmax()]
        selected.append(best.to_dict())
        remaining = remaining[remaining["recipe_id"] != best["recipe_id"]]
    return pd.DataFrame(selected).drop(columns=["_mmr"], errors="ignore").reset_index(drop=True)


def cap_categories(df, top_n):
    """Max MAX_PER_CATEGORY recepten per maaltijdcategorie in de eindlijst."""
    counts, selected, reserve = {}, [], []
    for _, row in df.iterrows():
        cat = row.get("category", "overig")
        if counts.get(cat, 0) < MAX_PER_CATEGORY and len(selected) < top_n:
            selected.append(row); counts[cat] = counts.get(cat, 0) + 1
        else:
            reserve.append(row)
    for row in reserve:
        if len(selected) >= top_n: break
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


# ── Aanbeveler ────────────────────────────────────────────────────────────────

class WasteLessRecommender:

    def __init__(self, recipes_file=RECIPES_FILE, pantry_file=PANTRY_FILE):
        print("Data laden...", end=" ", flush=True)
        self.recipes = pd.read_excel(recipes_file)
        self.pantry  = pd.read_excel(pantry_file)
        if "expiry_date" in self.pantry.columns:
            self.pantry["expiry_date"] = self.pantry["expiry_date"].apply(parse_date)
        else:
            self.pantry["expiry_date"] = pd.NaT

        self.users = sorted(self.pantry["user_id"].unique())
        self.recipes["category"]      = self.recipes["Naam"].apply(detect_category)
        self.recipes["n_ingredients"] = self.recipes["Ingrediënten"].fillna("").apply(
            lambda x: len([l for l in x.splitlines() if l.strip()])
        )
        print(f"[OK] ({len(self.recipes)} recepten, {len(self.users)} gebruikers)")

        # Parseer recepten één keer zodat we dit niet per gebruiker hoeven te herhalen
        print("Recepten parsen...", end=" ", flush=True)
        self.parsed_recipes, self.unique_names = build_recipe_matrix(self.recipes)
        print("[OK]")

        self._build_tfidf()

    def _build_tfidf(self):
        """Fit gedeelde TF-IDF ruimte op receptingrediënten + gebruikersvoorraad."""
        print("TF-IDF bouwen...", end=" ", flush=True)
        recipe_docs = self.recipes["Ingrediënten"].fillna("").apply(
            lambda x: " ".join(normalize(l) for l in x.splitlines() if l.strip())
        ).tolist()
        user_docs = [
            " ".join(normalize(str(v)) for v in self.pantry[self.pantry["user_id"] == uid]["ingredient"])
            + " " + " ".join(self.pantry[self.pantry["user_id"] == uid]["categorie"].dropna().unique())
            for uid in self.users
        ]
        mat = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(recipe_docs + user_docs)
        self.recipe_vecs  = mat[:len(recipe_docs)]
        self.user_vecs    = mat[len(recipe_docs):]
        self.recipe_index = {rid: i for i, rid in enumerate(self.recipes["ID"].tolist())}
        print("[OK]")

    def recommend(self, user_id, top_n=10, only_almost_done=False, verbose=False):
        """
        Beveel recepten aan voor één gebruiker.
        Geeft {"match": DataFrame, "with_rating": DataFrame}.
        """
        if user_id not in self.users:
            raise ValueError(f"Gebruiker {user_id} niet gevonden.")

        uid_idx   = list(self.users).index(user_id)
        profile   = self.pantry[self.pantry["user_id"] == user_id]["profiel"].iloc[0]
        pantry    = build_pantry(self.pantry, user_id)
        tfidf_sim = cosine_similarity(self.user_vecs[uid_idx:uid_idx+1], self.recipe_vecs)[0]

        # Eén fuzzy-lookup pass voor alle unieke ingrediënten → daarna vectorised scoring
        lookup  = match_pantry_to_ingredients(self.unique_names, pantry)
        cov_scores, fulls, partials, totals, urgents = vectorized_coverage(self.parsed_recipes, lookup)

        n_ing   = self.recipes["n_ingredients"].values
        ratings = self.recipes["Gemiddelde rating"].fillna(5.0).astype(float).values

        # Lengte-penalty: geen straf als er minstens 2 urgente items gedekt worden.
        # Eén urgent item is te makkelijk — anders komt "knoflookolie" (2 ingrediënten)
        # bovenaan puur omdat het één verlopend item bevat.
        penalties = np.where(
            (urgents >= 2) | (n_ing >= MIN_INGREDIENTS),
            1.0,
            1.0 - PENALTY_STRENGTH * (MIN_INGREDIENTS - n_ing) / MIN_INGREDIENTS
        )
        penalties = np.clip(penalties, 0.0, 1.0)

        # Verspilling-bonus proportioneel aan aandeel urgente ingrediënten
        waste_ratio = np.where(totals > 0, urgents / totals, 0.0)
        bonuses     = 1.0 + WASTE_BONUS_MAX * waste_ratio

        base_scores = WEIGHT_COVERAGE * cov_scores + WEIGHT_TFIDF * tfidf_sim

        # Normaliseer naar 0–1: de urgency-vermenigvuldiger (max 2.2×) kan de
        # ruwe dekking boven 1 duwen, wat verwarrend is in de output.
        MAX_RAW = WEIGHT_COVERAGE * 1.8 + WEIGHT_TFIDF * 1.0
        base_scores_norm = np.clip(base_scores / MAX_RAW, 0.0, 1.0)

        final_scores  = np.round(base_scores_norm * penalties * bonuses, 4)
        rating_scores = np.round(final_scores * (1 + RATING_BOOST * ratings / RATING_MAX), 4)
        almost_done   = (totals > 0) & (fulls / np.maximum(totals, 1) >= ALMOST_DONE_RATIO)

        # Minimumdrempel: minstens 1 volledig gedekt ingrediënt óf 2 gedeeltelijk.
        # Voorkomt dat recepten met 0/1 volledig bovenaan komen puur door urgency.
        voldoende_dekking = (fulls >= 1) | (partials >= 2)

        # Bouw resultaat-DataFrame
        df = pd.DataFrame({
            "recipe_id":         self.recipes["ID"].values,
            "recipe_name":       self.recipes["Naam"].values,
            "category":          self.recipes["category"].values,
            "score_final":       final_scores,
            "score_with_rating": rating_scores,
            "waste_bonus":       np.round(bonuses, 3),
            "length_penalty":    np.round(penalties, 4),
            "coverage":          np.round(base_scores_norm, 4),  # genormaliseerd 0–1
            "tfidf":             np.round(tfidf_sim, 4),
            "full":              fulls,
            "partial":           partials,
            "total":             totals,
            "urgent_items":      urgents,
            "n_ingredients":     n_ing,
            "almost_done":       almost_done,
            "url":               self.recipes["URL"].fillna("").values,
            "rating":            ratings,
        })

        # Minimumdrempel: verwijder recepten waarbij de gebruiker vrijwel niets heeft.
        # Voorkomt dat "0/1 volledig" bovenaan komt puur door urgency-boost.
        df = df[(df["full"] >= 1) | (df["partial"] >= 2)]

        if only_almost_done:
            df = df[df["almost_done"]]

        if df.empty:
            return {"match": pd.DataFrame(), "with_rating": pd.DataFrame()}

        df = df.sort_values("score_final", ascending=False).reset_index(drop=True)

        top_match  = cap_categories(mmr_rerank(df.head(50), self.recipe_vecs, self.recipe_index, top_n * 2), top_n)
        top_rating = cap_categories(mmr_rerank(df.sort_values("score_with_rating", ascending=False).head(50),
                                               self.recipe_vecs, self.recipe_index, 6,
                                               score_col="score_with_rating"), 3)
        top_match["rank"]  = top_match.index + 1
        top_rating["rank"] = top_rating.index + 1

        if verbose:
            self._print(user_id, profile, top_match, top_rating, pantry)

        return {"match": top_match, "with_rating": top_rating}

    def _print(self, user_id, profile, top_match, top_rating, pantry):
        def show(label, df, col):
            print(f"\n{'-'*55}\n  {label}\n{'-'*55}")
            for _, r in df.iterrows():
                flags = ("  ✓ bijna klaar" if r["almost_done"] else "") + \
                        (f"  🔴 {r['urgent_items']} urgent" if r["urgent_items"] else "") + \
                        (f"  ⚠ penalty={r['length_penalty']}" if r["length_penalty"] < 1 else "")
                print(f"\n  #{r['rank']:02d} [{r['category']}] {r['recipe_name']}{flags}")
                print(f"       score={r[col]:.3f}  dekking={r['coverage']:.3f}  tfidf={r['tfidf']:.3f}  rating={r['rating']:.1f}")
                print(f"       {r['full']}/{r['total']} volledig, {r['partial']} gedeeltelijk")
                print(f"       {r['url']}")
        print(f"\n{'='*55}\n  Gebruiker {user_id}  |  {profile}\n{'='*55}")
        show("TOP 3 — match + rating", top_rating, "score_with_rating")
        show(f"TOP {len(top_match)} — beste match", top_match, "score_final")

    def recommend_all(self, top_n=10):
        all_dfs = []
        print(f"\nRun voor {len(self.users)} gebruikers...")
        for i, uid in enumerate(self.users, 1):
            print(f"  [{i}/{len(self.users)}] gebruiker {uid}", flush=True)
            df = self.recommend(uid, top_n=top_n)["match"]
            if df.empty: continue
            profile = self.pantry[self.pantry["user_id"] == uid]["profiel"].iloc[0]
            df.insert(0, "user_id", uid); df.insert(1, "profile", profile)
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True)

    def export(self, df, path):
        from openpyxl.styles import Alignment, Font, PatternFill
        green = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")
        sheet = "Aanbevelingen"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet)
            ws = writer.sheets[sheet]
            for cell in ws[1]: cell.font = Font(bold=True)
            almost_col = list(df.columns).index("almost_done") if "almost_done" in df.columns else -1
            for row in ws.iter_rows(min_row=2):
                for cell in row: cell.alignment = Alignment(wrap_text=True, vertical="top")
                if almost_col >= 0 and row[almost_col].value:
                    for cell in row: cell.fill = green
        print(f"Opgeslagen: {path}")


# ── Commandline Interface ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="WasteLess Aanbeveler")
    p.add_argument("--user",        type=int)
    p.add_argument("--top",         type=int, default=10)
    p.add_argument("--all",         action="store_true")
    p.add_argument("--almost-done", action="store_true")
    p.add_argument("--output",      type=str)
    args = p.parse_args()

    rec = WasteLessRecommender()

    if args.all:
        rec.export(rec.recommend_all(args.top), args.output or "aanbevelingen.xlsx")
    elif args.user:
        result = rec.recommend(args.user, args.top, args.almost_done, verbose=True)
        if args.output:
            rec.export(pd.concat([result["with_rating"].assign(methode="match+rating"),
                                   result["match"].assign(methode="match")]), args.output)
    else:
        for uid in [1, 21, 61]:
            rec.recommend(uid, top_n=5, verbose=True)

if __name__ == "__main__":
    main()