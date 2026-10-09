#!/usr/bin/env python3
"""
seed_demo.py -- fill the database with SYNTHETIC demo properties so the app can be shown
before real listings are loaded.

THESE ARE NOT REAL LISTINGS. Every title, address and price below is made up for the demo,
and every row is stored with source='synthetic_demo' so the app can show a warning banner.
To use real data instead:  python3 -c "import db; db.load_properties_csv(db.connect('storage/advisor.db'), 'listings.csv')"

    python3 seed_demo.py --db storage/advisor.db
"""
import argparse

import db

# (title, city, district, type, price in IDR, land m2, building m2, bedrooms) -- all invented
DEMO = [
    ("Demo house A", "Jakarta", "Duren Sawit", "house", 1_800_000_000, 90, 70, 3),
    ("Demo house B", "Jakarta", "Jatinegara", "house", 2_400_000_000, 110, 90, 3),
    ("Demo townhouse C", "Jakarta", "Pulo Gadung", "townhouse", 1_450_000_000, 60, 60, 2),
    ("Demo house D", "Bekasi", "Jatiasih", "house", 1_150_000_000, 84, 60, 3),
    ("Demo house E", "Bekasi", "Bekasi Barat", "house", 1_650_000_000, 105, 80, 3),
    ("Demo house F", "Depok", "Sawangan", "house", 1_350_000_000, 100, 72, 3),
    ("Demo townhouse G", "Depok", "Beji", "townhouse", 980_000_000, 55, 50, 2),
    ("Demo house H", "Bandung", "Cibiru", "house", 1_250_000_000, 96, 72, 3),
    ("Demo house I", "Bandung", "Antapani", "house", 1_950_000_000, 120, 100, 4),
    ("Demo shophouse J", "Jakarta", "Matraman", "shophouse", 3_200_000_000, 75, 150, 0),
]

# Invented listing descriptions (SYNTHETIC, like everything above) so the TF-IDF recommender has text to work with.
DESCRIPTION = {
    "Demo house A": "Renovated family house in a quiet gated cluster, near schools and the toll road. "
                    "Carport for one car, small garden, new kitchen, 24-hour security.",
    "Demo house B": "Spacious two-storey family house close to Jatinegara station and the market. "
                    "Carport for two cars, garden, near schools, ready to move in.",
    "Demo townhouse C": "Compact townhouse in a gated cluster near the industrial estate and busway. "
                        "Low maintenance, 24-hour security, suits young couples or rental investment.",
    "Demo house D": "Affordable family house near the toll road and a shopping mall. "
                    "Carport, small garden, near schools, quiet residential street.",
    "Demo house E": "Family house in an established neighbourhood near the mall and commuter line station. "
                    "Carport for two cars, garden, renovated bathrooms.",
    "Demo house F": "Green suburban house with a large garden near a golf course and the toll road. "
                    "Carport, quiet cluster, good air, suits families with children.",
    "Demo townhouse G": "Budget townhouse near the university and the commuter line station. "
                        "Low maintenance, gated cluster, strong rental demand from students.",
    "Demo house H": "Family house with mountain views, cool climate, near the university campus. "
                    "Carport, garden, quiet street, needs minor renovation.",
    "Demo house I": "Large renovated family house near the city ring road, schools and a shopping mall. "
                    "Carport for two cars, big garden, four bedrooms.",
    "Demo shophouse J": "Three-storey shophouse on a busy main road near the busway and the market. "
                        "Ground floor retail space, offices above, high foot traffic, no garden.",
}

# APPROXIMATE district centres (lat, lon), not the address of any property. The demo properties have no real address.
DISTRICT_CENTRE = {
    "Duren Sawit": (-6.2300, 106.9100), "Jatinegara": (-6.2150, 106.8700), "Pulo Gadung": (-6.1850, 106.9100),
    "Jatiasih": (-6.2870, 106.9650), "Bekasi Barat": (-6.2300, 106.9900), "Sawangan": (-6.4000, 106.7700),
    "Beji": (-6.3800, 106.8250), "Cibiru": (-6.9200, 107.7200), "Antapani": (-6.9150, 107.6600),
    "Matraman": (-6.2050, 106.8600),
}


def seed(conn) -> int:
    existing = conn.execute("SELECT COUNT(*) FROM properties WHERE source = 'synthetic_demo'").fetchone()[0]
    added = 0
    if not existing:
        for t, city, dist, typ, price, land, bld, beds in DEMO:
            db.add_property(conn, title=t, city=city, district=dist, property_type=typ, price_idr=price,
                            land_m2=land, building_m2=bld, bedrooms=beds, source="synthetic_demo",
                            description=DESCRIPTION.get(t))
        added = len(DEMO)
    # fill coordinates on demo rows that lack them (also upgrades a database seeded before this column existed)
    for dist, (lat, lon) in DISTRICT_CENTRE.items():
        conn.execute("UPDATE properties SET latitude = ?, longitude = ? "
                     "WHERE source = 'synthetic_demo' AND district = ? AND latitude IS NULL", (lat, lon, dist))
    # fill descriptions on demo rows that lack them (databases seeded before the TF-IDF recommender)
    for title, text in DESCRIPTION.items():
        conn.execute("UPDATE properties SET description = ? "
                     "WHERE source = 'synthetic_demo' AND title = ? AND description IS NULL", (text, title))
    conn.commit()
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="storage/advisor.db")
    a = ap.parse_args()
    print(f"added {seed(db.connect(a.db))} synthetic demo properties")
