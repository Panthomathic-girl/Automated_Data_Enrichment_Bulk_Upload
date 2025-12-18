# generate_separate_files_us_full_52_fields.py
import json
import random
import uuid
from datetime import datetime

# === 100 REAL US HOSPITALS / HEALTH SYSTEMS / CLINICS ===
BASE_ACCOUNTS = [
    "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital", "Massachusetts General Hospital",
    "UCLA Medical Center", "Cedars-Sinai Medical Center", "NewYork-Presbyterian Hospital", "UCSF Medical Center",
    "Stanford Health Care", "Northwestern Memorial Hospital", "Mount Sinai Hospital", "Brigham and Women's Hospital",
    "Houston Methodist Hospital", "Barnes-Jewish Hospital", "Vanderbilt University Medical Center", "Duke University Hospital",
    "NYU Langone Hospitals", "Rush University Medical Center", "University of Michigan Hospitals", "Emory University Hospital",
    "Banner Health", "Kaiser Permanente", "Intermountain Healthcare", "Ascension Health", "Providence St. Joseph Health",
    "HCA Healthcare", "Tenet Healthcare", "Community Health Systems", "Trinity Health", "Advocate Aurora Health",
    "Children's Hospital of Philadelphia", "Boston Children's Hospital", "Texas Children's Hospital", "Cincinnati Children's Hospital",
    "Nationwide Children's Hospital", "Children's National Hospital", "Seattle Children's Hospital", "Lucile Packard Children's Hospital",
    "Banner Children's Specialists", "Phoenix Children's Hospital", "Cook Children's Medical Center", "Children's Mercy Kansas City",
    "Memorial Sloan Kettering Cancer Center", "MD Anderson Cancer Center", "Dana-Farber Cancer Institute",
    "Valley Hospital", "Methodist Hospital San Antonio", "Orlando Health", "Tampa General Hospital", "Jackson Memorial Hospital",
    "Montefiore Medical Center", "Hackensack University Medical Center", "Beaumont Hospital", "Henry Ford Hospital"
]

# === 50 MAJOR US CITIES ===
CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego",
    "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte", "Indianapolis",
    "San Francisco", "Seattle", "Denver", "Washington", "Boston", "El Paso", "Nashville", "Detroit",
    "Oklahoma City", "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore", "Milwaukee", "Albuquerque",
    "Tucson", "Fresno", "Mesa", "Sacramento", "Atlanta", "Kansas City", "Colorado Springs", "Omaha",
    "Raleigh", "Miami", "Long Beach", "Virginia Beach", "Oakland", "Minneapolis", "Tulsa", "Tampa",
    "Arlington", "New Orleans"
]

STATE_MAP = {city: state for city, state in zip(CITIES, [
    "NY","CA","IL","TX","AZ","PA","TX","CA","TX","CA","TX","FL","TX","OH","NC","IN",
    "CA","WA","CO","DC","MA","TX","TN","MI","OK","OR","NV","TN","KY","MD","WI","NM",
    "AZ","CA","AZ","CA","GA","MO","CO","NE","NC","FL","CA","VA","CA","MN","OK","FL","TX","LA"
])}

STATE_ABBREV = {
    "New York": "NY", "California": "CA", "Illinois": "IL", "Texas": "TX", "Arizona": "AZ",
    "Pennsylvania": "PA", "Florida": "FL", "Ohio": "OH", "North Carolina": "NC", "Indiana": "IN",
    "Washington": "WA", "Colorado": "CO", "District of Columbia": "DC", "Massachusetts": "MA",
    "Tennessee": "TN", "Michigan": "MI", "Oklahoma": "OK", "Oregon": "OR", "Nevada": "NV",
    "Kentucky": "KY", "Maryland": "MD", "Wisconsin": "WI", "New Mexico": "NM", "Georgia": "GA",
    "Missouri": "MO", "Nebraska": "NE", "Virginia": "VA", "Minnesota": "MN", "Louisiana": "LA"
}

SUFFIXES = ["", " Medical Center", " Hospital", " Health System", " Clinic", " Specialists",
            " Children's Hospital", " Cancer Center", " - Main Campus", " University Medical Center"]

def random_phone():
    return random.randint(1000000000, 9999999999) if random.random() > 0.15 else None

def random_date(start_year, end_year):
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}T00:00:00.000Z"

def generate_street():
    streets = ["Main St", "Broadway", "Elm St", "Oak Ave", "Cedar Ln", "Maple Dr", "Washington Blvd",
               "Park Ave", "Sunset Blvd", "Dobson Rd", "Central Ave", "University Dr"]
    number = random.randint(100, 9999)
    street = random.choice(streets)
    suite = random.choice(["", f"\nSuite {random.randint(100, 510)}", f"\nSte {random.randint(1, 20)}"])
    return f"{number} {street}{suite}".strip()

def generate_zip():
    return random.randint(10000, 99999)

# === ALL 52 FIELDS IN EXACT ORDER (as per your example) ===
FIELD_ORDER = [
    "id", "is_active", "created_at", "is_deleted", "record_id", "account_owner_id",
    "account_name", "account_type", "parent_account_id", "website", "phone", "email",
    "billing_street", "billing_city", "billing_state", "billing_code", "billing_country",
    "shipping_street", "shipping_city", "shipping_state", "shipping_code", "shipping_country",
    "ship_to_address", "ship_to_city", "ship_to_postal_code", "shipped_to_state", "billed_to_state",
    "territories", "idn_parent", "group_purchasing_organization", "price_list", "net_payment_terms",
    "potential_value", "primary_contact_name", "designation", "alternate_phone",
    "dea_lic_num", "dea_expiration", "pharmacy_lic_num", "pharmacy_expiration",
    "medical_license_number", "medical_license_expiration_date", "npi_lic_num", "npi_expiration",
    "primary_specialty", "industry", "product_interest", "onboarding_date", "lead_source",
    "notes", "accountTypeId", "lead_stage"
]

def generate_record():
    base = random.choice(BASE_ACCOUNTS)
    city = random.choice(CITIES)
    suffix = random.choice(SUFFIXES)
    name = f"{base}{suffix} {city}".strip() if random.random() > 0.3 else f"{base}{suffix}".strip()

    state_full = STATE_MAP[city]
    state = STATE_ABBREV.get(state_full, state_full)
    street = generate_street()
    zip_code = generate_zip()

    record = {
        "id": str(uuid.uuid4()),
        "is_active": True,
        "created_at": "2025-12-15T05:04:52.749Z",
        "is_deleted": False,
        "record_id": f"zcrm_55865590000{random.randint(5000000, 9999999):07d}",
        "account_owner_id": "zcrm_5586559000001026001",
        "account_name": name,
        "account_type": random.choice(["Hospital", "Clinic", "Health System", "Children's Hospital", "Cancer Center"]),
        "parent_account_id": random.choice([None, random.choice(BASE_ACCOUNTS) + " Health"]),
        "website": None if random.random() > 0.4 else f"www.{name.lower().replace(' ', '').replace('-', '')}.com",
        "phone": random_phone(),
        "email": None if random.random() > 0.5 else f"info@{name.lower().replace(' ', '')}.com",
        "billing_street": street,
        "billing_city": city,
        "billing_state": state,
        "billing_code": zip_code,
        "billing_country": "United States",
        "shipping_street": None,
        "shipping_city": None,
        "shipping_state": None,
        "shipping_code": None,
        "shipping_country": "United States",
        "ship_to_address": street,
        "ship_to_city": city,
        "ship_to_postal_code": zip_code,
        "shipped_to_state": state,
        "billed_to_state": state,
        "territories": None,
        "idn_parent": random.choice([None, "Banner Health", "Kaiser Permanente", "HCA Healthcare", "Ascension"]),
        "group_purchasing_organization": None,
        "price_list": random.choice(["Hospital Price List", "Clinic Price List", None]),
        "net_payment_terms": random.choice([30, 45, 60]),
        "potential_value": None,
        "primary_contact_name": None,
        "designation": None,
        "alternate_phone": None,
        "dea_lic_num": None if random.random() > 0.3 else f"{random.choice(['A','B','F','M'])}{random.randint(1000000,9999999)}",
        "dea_expiration": random_date(2025, 2028) if random.random() > 0.7 else None,
        "pharmacy_lic_num": None,
        "pharmacy_expiration": None,
        "medical_license_number": None if random.random() > 0.4 else f"{random.choice(['Y','R'])}{random.randint(100000,999999)}",
        "medical_license_expiration_date": random_date(2010, 2025) if random.random() > 0.8 else None,
        "npi_lic_num": None,
        "npi_expiration": None,
        "primary_specialty": random.choice(["Pediatrics", "Cardiology", "Oncology", "General", "Orthopedics", None]),
        "industry": None,
        "product_interest": None,
        "onboarding_date": random_date(2020, 2024),
        "lead_source": random.choice(["Banner", "Referral", "Website", "Trade Show", None]),
        "notes": None,
        "accountTypeId": str(uuid.uuid4()),
        "lead_stage": random.choice(["UN_VERIFIED", "VERIFIED", "QUALIFIED"])
    }

    # Ensure ALL 52 fields are present, in order
    ordered_record = {field: record.get(field, None) for field in FIELD_ORDER}
    return ordered_record

def generate_leads(records_sample, n=1000):
    leads = []
    variations = [
        lambda r: {"account_name": r["account_name"], "billing_city": r["billing_city"], "phone": r["phone"]},
        lambda r: {"account_name": r["account_name"].replace("Hospital", "Medical Center"), "website": r["website"]},
        lambda r: {"account_name": r["account_name"] + " Specialists", "email": r["email"]},
        lambda r: {"account_name": r["account_name"].replace("Children's", "Childrens"), "billing_state": r["billing_state"]},
        lambda r: {"account_name": r["account_name"], "billing_street": r["billing_street"]},
    ]

    for i in range(n):
        base = random.choice(records_sample)
        variation = random.choice(variations)(base)

        lead = {
            "id": str(uuid.uuid4()),
            "is_active": True,
            "created_at": "2025-12-15T05:04:52.749Z",
            "is_deleted": False,
            "record_id": f"zcrm_55865590000{random.randint(1000000, 1999999):07d}",
            "account_owner_id": "zcrm_5586559000001026001",
            "lead_stage": "UN_VERIFIED",
            **variation  # only a few fields + variations
        }
        leads.append(lead)

    return leads

# === GENERATE & SAVE ===
random.seed(42)

print("Generating 10,000 master records with ALL 52 fields...")
records = [generate_record() for _ in range(10000)]

print("Generating 1,000 incoming leads (minimal fields + variations for deduplication)...")
leads = generate_leads(records[:500], 1000)

# Save
with open("us_records_10000.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

with open("us_leads_1000.json", "w", encoding="utf-8") as f:
    json.dump(leads, f, indent=2, ensure_ascii=False)

print("\nDONE!")
print("us_records_10000.json → 10,000 records with EXACTLY 52 fields each")
print("us_leads_1000.json            → 1,000 incoming leads with partial data for testing enrichment/dedup")
print("\nTest with curl:")
print("curl -X POST http://127.0.0.1:8000/enrichment/process \\")
print("  -H 'Content-Type: application/json' \\")
print("  -d '{\"records\": $(cat us_records_10000_full_52_fields.json), \"leads\": $(cat us_leads_1000_minimal.json)}' | jq .")