# WasteLess

Dit is onze content-based recept-aanbeveler die je voorraad en houdbaarheidsdatums gebruikt om recepten voor te stellen, zodat je minder eten weggooit.

Gemaakt voor het vak Recommender Systems aan de Universiteit van Amsterdam.

---

## Wat het doet

WasteLess beveelt recepten aan op basis van wat je al in huis hebt. Ingredienten die bijna verlopen staan hoger in de ranking, zodat je ze op tijd gebruikt. Recepten komen van Albert Heijn Allerhande.

---

## Hoe het werkt

1. Ingredienten uit recepten en de voorraad worden genormaliseerd met een Nederlandse Snowball-stemmer
2. Een hash map koppelt voorraadartikelen aan recept-ingredienten
3. Recepten krijgen een score op basis van voorraaddekking (75%) en TF-IDF-similariteit (25%)
4. Een urgentiemultiplier (tot 2.2x) geeft een boost aan ingredienten die bijna verlopen
5. MMR-herrangschikking en een categorielimiet zorgen voor gevarieerde aanbevelingen

---

## Bestanden

| Bestand | Beschrijving |
|---|---|
| `wasteless_recommender.ipynb` | Hoofd-notebook van de aanbeveler |
| `ah_recepten_scraper_en_pantry_builder.ipynb` | Scraper voor AH Allerhande + pantry-proxygenerator |
| `recepten_AH.xlsx` | Dataset met 2000 gescrapete recepten |
| `pantry_proxies.xlsx` | Synthetische voorraaddata voor 100 gebruikers |

---

## Installatie

```bash
pip install requests beautifulsoup4 openpyxl pandas scikit-learn nltk numpy
```

Voer eerst `ah_recepten_scraper_en_pantry_builder.ipynb` uit om de data te genereren, daarna `wasteless_recommender.ipynb`.

Voor een kortere testrun verlaag je `AANTAL_RECEPTEN` in de configuratiecel (standaard 2000, begin met 50-200).

---

## Data

Recepten worden gescraped van Albert Heijn Allerhande via een anoniem API-token. Voorraaddata is synthetisch, gegenereerd voor 100 gebruikers verdeeld over 5 huishoudprofielen: gezin, student, stel, alleenstaande en gezondheidsgerichte eter.

---

## Auteurs

Diaz Wiersma, Sebastian Rociu, Oliver Martin, Liam Boesten

Vak: Recommender Systems (5072RESY6Y), Universiteit van Amsterdam

Begeleiders: Simon Pauw, Dina Strikovic