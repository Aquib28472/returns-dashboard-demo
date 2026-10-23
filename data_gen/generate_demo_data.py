"""Generate synthetic source tables for the Returns Dashboard demo.

Everything here is invented: customers, products, owners, orders, documents, tracking numbers.
Each scenario below is built to land in one branch of the model's return classification, and
expected_actions.csv records the owner and action group each open line was designed to get, so the
model's output can be checked after a refresh.

All dates are relative to --today (default 2026-10-23), so the same arguments always give the
same files. The report itself ages lines against the real current date (TODAY()).
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

WHR, RET = "WHR", "RET"  # warehouse receipt of a return / return delivery dropped (dispatched)

PLANT_BU = {"FLD1": "DIV-A", "FLD2": "DIV-A", "CON1": "DIV-A", "FLD3": "DIV-B", "CON2": "DIV-B"}
FIELD_PLANTS = ["FLD1", "FLD2", "FLD3"]
CONS_PLANTS = ["CON1", "CON2"]
DIV_B_CODE = "10"  # product division that maps to DIV-B; "20" and "30" map to DIV-A
DCS = ["DC01", "DC02", "DC03", "DC04"]

OWNERS = ["MARLOW QUINN", "TESSA BRANDT", "RAVI OKAFOR", "NOOR CASTELLAN", "ELIAS VARGA",
          "MINA HALLORAN", "JONAS PRYCE", "INES DUARTE", "CALLUM REYES", "SANA WHITLOCK"]
REGION_NAMES = ["North", "South", "East", "West", "Central"]
CUST_A = ["Northwind", "Harbor", "Summit", "Lakeside", "Pioneer", "Cedar", "Bluebird", "Granite", "Maple",
          "Riverbend", "Oakridge", "Silverline", "Kestrel", "Foxglove", "Ironwood"]
CUST_B = ["Clinics", "Health", "Care Group", "Medical Center", "Hospital", "Surgical", "Labs", "Supply Co"]
CITIES = [("Springfield", "IL", "627"), ("Riverton", "WY", "825"), ("Fairview", "OR", "970"), ("Georgetown", "TX", "786"),
          ("Madison", "WI", "537"), ("Clinton", "IA", "527"), ("Ashland", "KY", "411"), ("Salem", "OR", "973")]
PRODUCT_KINDS = ["Catheter", "Guidewire", "Sheath", "Balloon", "Stent graft", "Introducer", "Syringe set", "Valve kit"]
PRODUCT_CLASSES = ["Access", "Delivery systems", "Implants", "Accessories"]
PRODUCT_GROUPS = ["Core range", "Specialty", "Legacy", "Starter kits"]
REASONS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11", "F12", "F13", "R14", "R26", ""]
REASON_P = [0.16, 0.05, 0.06, 0.08, 0.07, 0.05, 0.08, 0.03, 0.09, 0.03, 0.03, 0.08, 0.03, 0.04, 0.10, 0.02]
CARRIER_SEQ = [("Label created", "Shipment information sent to carrier"),
               ("Picked up", "Picked up"),
               ("In transit", "In transit"),
               ("Out for delivery", "On vehicle for delivery"),
               ("Delivered", "Delivered")]

# scenario -> (kind, plant pool, age range, cockpit mode, carrier mode, movement mode, has tracking, expected owner, expected group)
# kind: N = no delivery on the return order and no plant/batch/serial yet, A = delivery not yet created,
#       B = delivery exists. Cockpit modes: same / other / transit / restricted / None.
SCENARIOS = {
    "NODEL_NOTRACK":          ("N", "any",   (3, 280),  None,         None,        None,      False, "Inventory", "2"),
    "NODEL_DELIVERED":        ("N", "any",   (5, 120),  None,         "delivered", None,      True,  "Inventory", "1"),
    "NODEL_MOVING":           ("N", "any",   (3, 60),   None,         "moving",    None,      True,  "Inventory", "1"),
    "NODEL_LABEL_FAR":        ("N", "field", (3, 90),   None,         None,        None,      True,  "Inventory", "1"),
    "NODEL_LABEL_CONS":       ("N", "cons",  (3, 90),   None,         None,        None,      True,  "Inventory", "4"),
    "CONS_SHIPPED":           ("A", "cons",  (3, 60),   None,         "moving",    None,      True,  "Inventory", "1"),
    "LABEL_NOT_DROPPED":      ("A", "field", (3, 90),   None,         "label",     None,      True,  "Inventory", "1"),
    "NOT_DROPPED_30":         ("A", "any",   (31, 290), "rand_a",     None,        None,      False, "Inventory", "1"),
    "A_YOUNG_NO_MOVE":        ("A", "any",   (2, 30),   None,         None,        None,      False, "Rep",       "4"),
    "A_STILL_IN_FIELD":       ("A", "any",   (2, 30),   "same",       None,        None,      False, "Rep",       "4"),
    "A_BACK_IN_FIELD":        ("A", "divA",  (2, 30),   "other",      None,        None,      False, "Inventory", "3"),
    "DROPPED_NEVER_REP":      ("B", "any",   (3, 200),  "same",       None,        "ret",     False, "Rep",       "4"),
    "DROPPED_NEVER_REP_NOCK": ("B", "any",   (3, 200),  None,         None,        "ret",     False, "Rep",       "4"),
    "DROPPED_NEVER_CARRIER":  ("B", "any",   (3, 200),  None,         "moving",    "ret",     True,  "Carrier",   "5"),
    "DROPPED_NEVER_CARRIER_ACTIVE": ("B", "any", (3, 200), "same",    "moving",    "ret",     True,  "Carrier",   "5"),
    "DROPPED_NEVER_WH":       ("B", "any",   (3, 200),  "transit",    "delivered", "ret",     True,  "Warehouse", "6"),
    "UNRESOLVED_REP":         ("B", "any",   (3, 200),  "transit",    None,        None,      False, "Rep",       "4"),
    "UNRESOLVED_CARRIER":     ("B", "any",   (3, 200),  "transit",    "moving",    None,      True,  "Carrier",   "5"),
    "UNRESOLVED_WH":          ("B", "any",   (3, 200),  "transit",    "delivered", None,      True,  "Warehouse", "6"),
    "CONFLICT_REVIEW":        ("B", "any",   (3, 150),  None,         "delivered", None,      True,  "Inventory", "1"),
    "ACTIVE_DELIVERED_REVIEW": ("B", "any",  (3, 150),  "same",       "delivered", None,      True,  "Inventory", "1"),
    "ACTIVE_CARRIER":         ("B", "any",   (3, 150),  "same",       "moving",    None,      True,  "Carrier",   "5"),
    "ACTIVE_REP":             ("B", "any",   (3, 150),  "same",       "label",     None,      True,  "Rep",       "4"),
    "PARTIAL":                ("B", "any",   (6, 200),  "rand_b",     None,        "partial", False, "Inventory", "3"),
    "CONFIRMED_CLOSE":        ("B", "any",   (6, 250),  "rand_c",     None,        "full",    False, "Inventory", "7"),
    "CONFIRMED_STILL":        ("B", "any",   (6, 250),  "same",       None,        "full",    False, "Inventory", "3"),
    "INBOUNDED_CLOSE_NOQTY":  ("B", "any",   (6, 250),  None,         None,        "full0",   False, "Inventory", "7"),
    "NO_MOVE_DELIVERY":       ("B", "any",   (3, 250),  None,         None,        None,      False, "Rep",       "4"),
}
OPEN_WEIGHTS = {
    "NODEL_NOTRACK": 55, "NODEL_DELIVERED": 14, "NODEL_MOVING": 18, "NODEL_LABEL_FAR": 22, "NODEL_LABEL_CONS": 12,
    "CONS_SHIPPED": 16, "LABEL_NOT_DROPPED": 24, "NOT_DROPPED_30": 34, "A_YOUNG_NO_MOVE": 22, "A_STILL_IN_FIELD": 14,
    "A_BACK_IN_FIELD": 8, "DROPPED_NEVER_REP": 40, "DROPPED_NEVER_REP_NOCK": 20, "DROPPED_NEVER_CARRIER": 30,
    "DROPPED_NEVER_CARRIER_ACTIVE": 8, "DROPPED_NEVER_WH": 22, "UNRESOLVED_REP": 20, "UNRESOLVED_CARRIER": 8,
    "UNRESOLVED_WH": 10, "CONFLICT_REVIEW": 14, "ACTIVE_DELIVERED_REVIEW": 8, "ACTIVE_CARRIER": 6, "ACTIVE_REP": 8,
    "PARTIAL": 18, "CONFIRMED_CLOSE": 60, "CONFIRMED_STILL": 20, "INBOUNDED_CLOSE_NOQTY": 12, "NO_MOVE_DELIVERY": 40,
}


def digits(rng, n: int) -> str:
    return "".join(str(d) for d in rng.integers(0, 10, n))


def build(seed: int = 11, today: date = date(2026, 10, 23)):
    rng = np.random.default_rng(seed)
    year_start = date(2026, 1, 1)

    # ---- products --------------------------------------------------------------------------
    products = []
    for i in range(44):
        div = DIV_B_CODE if i % 4 == 0 else ("20" if i % 2 else "30")
        products.append({"sku": str(500100 + i * 37), "div": div,
                         "desc": f"{PRODUCT_KINDS[i % len(PRODUCT_KINDS)]} {3 + (i % 9)}F {60 + (i * 7) % 140} cm",
                         "class": PRODUCT_CLASSES[i % 4], "group": PRODUCT_GROUPS[(i // 2) % 4]})

    # ---- customers and field accounts -------------------------------------------------------
    n_reg = {"A": 4, "B": 3}
    n_terr = {"A": 14, "B": 9}

    def geo(prefix: str, t: int):
        r = (t - 1) % n_reg[prefix] + 1
        return {"region": f"{prefix}-R{r}", "terr": f"{prefix}-T{t:02d}", "area": f"Area {prefix}{r}"}

    customers = []
    for i in range(170):
        name = f"{CUST_A[i % len(CUST_A)]} {CUST_B[(i // len(CUST_A)) % len(CUST_B)]}"
        if i >= len(CUST_A) * len(CUST_B):
            name += f" {i}"
        city, st, z3 = CITIES[i % len(CITIES)]
        customers.append({
            "no": "2" + digits(rng, 9), "sold_to": "1" + digits(rng, 9), "name": name,
            "A": geo("A", int(rng.integers(1, 15))), "B": geo("B", int(rng.integers(1, 10))),
            "city": city, "state": st, "zip": z3 + digits(rng, 2), "phone": f"555-01{digits(rng, 2)}",
            "email": f"orders{i}@example.com", "addr": f"{int(rng.integers(10, 999))} {CUST_A[(i * 3) % len(CUST_A)]} Road",
        })
    field_accounts = {"A": [], "B": []}
    for prefix in ("A", "B"):
        for t in range(1, n_terr[prefix] + 1):
            g = geo(prefix, t)
            no = "80000" + digits(rng, 5)
            field_accounts[prefix].append({"no": no, "sold_to": no, "name": f"Field stock {prefix}-T{t:02d}",
                                           "A": g if prefix == "A" else {"region": "", "terr": "", "area": ""},
                                           "B": g if prefix == "B" else {"region": "", "terr": "", "area": ""},
                                           "city": "", "state": "", "zip": "", "phone": "", "email": "", "addr": ""})
    # customers that are covered by an account-level alignment row only
    acct_customers = [customers[i] for i in (3, 17, 31, 59, 88, 120)]

    # ---- owner alignment -----------------------------------------------------------------------
    align, oi = [], 0
    for bu, prefix in (("DIV-A", "A"), ("DIV-B", "B")):
        for r in range(1, n_reg[prefix] + 1):
            owner = OWNERS[oi % len(OWNERS)]
            consign = owner + ("_OPEN" if oi % 5 == 4 else "")
            oi += 1
            for t in range(1, n_terr[prefix] + 1):
                if (t - 1) % n_reg[prefix] != r - 1:
                    continue
                align.append({"BU": bu, "AREA": f"Area {prefix}{r}", "Region ID": f"{prefix}-R{r}",
                              "Region Name": f"{REGION_NAMES[(r - 1) % 5]} ({prefix}{r})",
                              "Region Manager": f"Manager {prefix}{r}", "Territory ID": f"{prefix}-T{t:02d}",
                              "Territory Name": f"Territory {prefix}{t:02d}", "Field Associate": f"Associate {prefix}{t:02d}",
                              "Email Address": f"associate.{prefix.lower()}{t:02d}@example.com", "Acct #": "",
                              "OWNER - REP": owner, "OWNER - CONSIGN": consign, "REP": owner, "CONSIGN": consign})
    for k, c in enumerate(acct_customers):
        for bu in ("DIV-A", "DIV-B"):
            owner = OWNERS[(k + 3) % len(OWNERS)]
            align.append({"BU": bu, "AREA": "", "Region ID": "", "Region Name": "", "Region Manager": "", "Territory ID": "",
                          "Territory Name": "", "Field Associate": "", "Email Address": "", "Acct #": c["no"],
                          "OWNER - REP": owner, "OWNER - CONSIGN": owner, "REP": owner, "CONSIGN": owner})
    alignment = pd.DataFrame(align)
    acct_nos = {c["no"] for c in acct_customers}

    # ---- builders ------------------------------------------------------------------------------
    delivery, movement, cockpit, carrier, expected = [], [], [], [], []
    used = {"serial": set(), "so": set(), "track": set()}

    def uniq(kind: str, make):
        while True:
            v = make()
            if v not in used[kind]:
                used[kind].add(v)
                return v

    def pick_plant(pool: str):
        if pool == "field":
            return FIELD_PLANTS[int(rng.choice(3, p=[0.38, 0.34, 0.28]))]
        if pool == "cons":
            return CONS_PLANTS[int(rng.choice(2, p=[0.6, 0.4]))]
        if pool == "divA":
            return ["FLD1", "FLD2", "CON1"][int(rng.integers(0, 3))]
        return (FIELD_PLANTS + CONS_PLANTS)[int(rng.choice(5, p=[0.26, 0.22, 0.17, 0.19, 0.16]))]

    def pick_product(bu: str):
        pool = [p for p in products if (p["div"] == DIV_B_CODE) == (bu == "DIV-B")]
        return pool[int(rng.integers(0, len(pool)))]

    def geo_for(entity, prefix: str, order_idx: int):
        """Region/territory as the territory hierarchy would return them, with a few deliberate gaps."""
        g = entity[prefix]
        if entity["no"] in acct_nos and order_idx % 2 == 0:
            return "X-R9", "X-T99"           # only the account-level alignment row can match
        u = rng.random()
        if u < 0.80:
            return g["region"], g["terr"]
        if u < 0.89:
            return "X-R9", g["terr"]         # region unknown, territory known
        if u < 0.95:
            return "X-R9", "X-T99"           # nothing matches -> default owner
        return g["region"], g["terr"]

    def carrier_rows(so, po, tids, mode, order_dt):
        n_steps = {"label": 1, "moving": 3, "delivered": 5}[mode]
        for tid in tids:
            start = order_dt + timedelta(days=int(rng.integers(1, 4)))
            party = ""
            if rng.random() < 0.25:
                party = {"label": "Rep", "moving": "Carrier", "delivered": "Warehouse"}[mode]
            dt = start
            for step in range(n_steps):
                if step:
                    dt = dt + timedelta(days=int(rng.integers(1, 3)))   # events move forward in time
                dt = min(today, dt)
                overall, detail = CARRIER_SEQ[step]
                carrier.append({"tracking_id": tid, "reference": so, "po": po, "event_date": dt, "status": detail,
                                "responsible_party": party if step == n_steps - 1 else "", "overall_status": overall,
                                "location": CITIES[int(rng.integers(0, len(CITIES)))][0]})

    def add_open_line(scn: str, line_no: int = 10, so: str | None = None):
        kind, pool, (lo, hi), ck, cr, mvm, has_trk, exp_owner, exp_group = SCENARIOS[scn]
        plant = pick_plant(pool)
        bu = PLANT_BU[plant]
        prefix = "A" if bu == "DIV-A" else "B"
        is_field = plant in FIELD_PLANTS
        if kind == "N" and pool == "any":
            is_field = rng.random() < 0.5
            plant = (FIELD_PLANTS if is_field else CONS_PLANTS)[0]
            bu = "DIV-B" if rng.random() < 0.3 else "DIV-A"
            prefix = "A" if bu == "DIV-A" else "B"
        if kind == "N" and pool == "field":
            is_field = True
        if kind == "N" and pool == "cons":
            is_field = False
        entity = (field_accounts[prefix][int(rng.integers(0, len(field_accounts[prefix])))] if is_field
                  else customers[int(rng.integers(0, len(customers)))])
        prod = pick_product(bu)
        age = int(rng.integers(lo, min(hi, (today - year_start).days) + 1))
        order_dt = today - timedelta(days=age)
        so = so or uniq("so", lambda: "5" + digits(rng, 9))
        po = "PO" + digits(rng, 7)
        sku = prod["sku"].zfill(18) if rng.random() < 0.5 else prod["sku"]
        batch = "L" + digits(rng, 7)
        serial = uniq("serial", lambda: digits(rng, 9))
        n = int(rng.integers(2, 7)) if mvm == "partial" else int(rng.integers(1, 7))
        expiry = today + timedelta(days=int(rng.integers(-70, 600)))
        order_reason = str(rng.choice(REASONS, p=REASON_P))
        blank3 = kind == "N"
        branch = "" if blank3 else plant   # the model's plant checks compare the whole branch value
        order_idx = len(delivery)
        region, terr = geo_for(entity, prefix, order_idx)
        ds = "A" if kind in ("N", "A") else "B"
        delivery_no = "" if kind in ("N", "A") else "08" + digits(rng, 8)
        delivery_dt = None if kind in ("N", "A") else min(today, order_dt + timedelta(days=int(rng.integers(0, 6))))
        tids = [uniq("track", lambda: digits(rng, 12)) for _ in range(int(rng.integers(1, 3)))] if has_trk else []
        qty = 0 if mvm == "full0" else n
        delivery.append({
            "order_status": "Not Closed", "date": delivery_dt, "so_overall_delivery_status": ds, "order_date": order_dt,
            "sales_organization": "1000", "branch": branch, "product_division": prod["div"],
            "po_type_desc": "Field return", "sales_order": so, "so_line": line_no, "sales_order_type_desc": "Field Inv Pickup",
            "order_reason": order_reason, "delivery_no": delivery_no,
            "delivery_line": None if kind in ("N", "A") else line_no, "expiry": None if (blank3 or rng.random() < 0.18) else expiry,
            "batch": "" if blank3 else batch, "serial": "" if blank3 else serial, "po_no": po,
            "shipping_plant": DCS[int(rng.integers(0, 4))], "ship_to_no": entity["no"], "ship_to": entity["name"],
            "sold_to": entity["sold_to"], "sku": sku, "description": prod["desc"], "product_class": prod["class"],
            "product_group": prod["group"], "area_a": entity["A"]["area"], "area_b": entity["B"]["area"],
            "region_a": entity["A"]["region"], "region_b": entity["B"]["region"],
            "territory_a": entity["A"]["terr"], "territory_b": entity["B"]["terr"],
            "tracking": ", ".join(tids), "region_id": region, "territory_id": terr, "return_order_qty": qty,
        })
        if cr:
            carrier_rows(so, po, tids, cr, order_dt)
        # field inventory position for this unit
        mode = ck
        if mode and mode.startswith("rand_"):
            options = {"rand_a": [None, None, "same", "restricted"], "rand_b": [None, "same"], "rand_c": [None, "restricted"]}[mode]
            mode = options[int(rng.integers(0, len(options)))]
        if mode and not blank3:
            holder = entity
            if mode == "other":
                holder = customers[int(rng.integers(0, len(customers)))]
                while holder["no"] == entity["no"]:
                    holder = customers[int(rng.integers(0, len(customers)))]
            q = {"same": (n, 0, 0, n), "other": (n, 0, 0, n), "transit": (0, 0, n, 0), "restricted": (0, n, 0, 0)}[mode]
            cockpit.append({"branch": f"{plant}_{int(rng.integers(1, 9)):02d}", "customer_no": holder["no"], "sku": prod["sku"].zfill(18),
                            "description": prod["desc"], "batch": batch, "vendor_batch": "", "serial": serial,
                            "expiry": expiry, "customer_name": holder["name"], "qty_on_hand": q[0], "qty_restricted": q[1],
                            "qty_in_transit": q[2], "qty_total": q[3]})
        # warehouse movements for this delivery
        if mvm and delivery_no:
            ret_dt = min(today, delivery_dt + timedelta(days=int(rng.integers(0, 3))))
            movement.append({"movement_type": RET, "posting_date": ret_dt, "reference": delivery_no, "sku": sku,
                             "description": prod["desc"], "matdoc_no": "49" + digits(rng, 8), "sales_order": so, "batch": batch,
                             "plant": plant, "product_division": prod["div"], "quantity": -n})
            if mvm in ("partial", "full", "full0"):
                got = int(rng.integers(1, n)) if mvm == "partial" else n
                movement.append({"movement_type": WHR, "posting_date": min(today, ret_dt + timedelta(days=int(rng.integers(2, 9)))),
                                 "reference": delivery_no, "sku": sku, "description": prod["desc"], "matdoc_no": "50" + digits(rng, 8),
                                 "sales_order": "", "batch": batch, "plant": DCS[int(rng.integers(0, 4))],
                                 "product_division": prod["div"], "quantity": got})
        expected.append({"sales_order": so, "so_line": line_no, "scenario": scn, "expected_owner": exp_owner, "expected_group": exp_group})
        return so

    for scn, w in OPEN_WEIGHTS.items():
        for _ in range(w):
            add_open_line(scn)

    # ---- history: closed and received returns (feed the monthly performance page) ------------------
    def add_history(kind: str):
        plant = pick_plant("any")
        bu = PLANT_BU[plant]
        prefix = "A" if bu == "DIV-A" else "B"
        is_field = plant in FIELD_PLANTS
        entity = (field_accounts[prefix][int(rng.integers(0, len(field_accounts[prefix])))] if is_field
                  else customers[int(rng.integers(0, len(customers)))])
        prod = pick_product(bu)
        so = uniq("so", lambda: "5" + digits(rng, 9))
        po = "PO" + digits(rng, 7)
        order_dt = year_start + timedelta(days=int(rng.integers(0, (today - year_start).days - 20)))
        sku = prod["sku"]
        batch = "L" + digits(rng, 7)
        serial = uniq("serial", lambda: digits(rng, 9))
        n = int(rng.integers(1, 7))
        region, terr = geo_for(entity, prefix, len(delivery))
        has_delivery = kind != "nodel"
        gap = {"ontime": int(rng.integers(0, 2)), "late": int(rng.integers(2, 15))}.get(kind, 0)
        delivery_dt = min(today, order_dt + timedelta(days=gap)) if has_delivery else None
        delivery_no = "08" + digits(rng, 8) if has_delivery else ""
        status = {"ontime": "Closed", "late": "Closed", "nodel": "Closed", "nc_received": "Not Closed", "nc_pending": "Not Closed"}[kind]
        ds = "C" if has_delivery else "A"
        delivery.append({
            "order_status": status, "date": delivery_dt, "so_overall_delivery_status": ds, "order_date": order_dt,
            "sales_organization": "1000", "branch": plant, "product_division": prod["div"],
            "po_type_desc": "Field return", "sales_order": so, "so_line": 10, "sales_order_type_desc": "Field Inv Pickup",
            "order_reason": str(rng.choice(REASONS, p=REASON_P)), "delivery_no": delivery_no,
            "delivery_line": 10 if has_delivery else None, "expiry": today + timedelta(days=int(rng.integers(-70, 600))),
            "batch": batch, "serial": serial, "po_no": po, "shipping_plant": DCS[int(rng.integers(0, 4))],
            "ship_to_no": entity["no"], "ship_to": entity["name"], "sold_to": entity["sold_to"], "sku": sku,
            "description": prod["desc"], "product_class": prod["class"], "product_group": prod["group"],
            "area_a": entity["A"]["area"], "area_b": entity["B"]["area"], "region_a": entity["A"]["region"],
            "region_b": entity["B"]["region"], "territory_a": entity["A"]["terr"], "territory_b": entity["B"]["terr"],
            "tracking": "", "region_id": region, "territory_id": terr, "return_order_qty": n,
        })
        if has_delivery:
            ret_dt = delivery_dt
            movement.append({"movement_type": RET, "posting_date": ret_dt, "reference": delivery_no, "sku": sku,
                             "description": prod["desc"], "matdoc_no": "49" + digits(rng, 8), "sales_order": so, "batch": batch,
                             "plant": plant, "product_division": prod["div"], "quantity": -n})
            if kind == "nc_received" or (kind != "nc_pending" and rng.random() < 0.7):
                movement.append({"movement_type": WHR, "posting_date": min(today, ret_dt + timedelta(days=int(rng.integers(2, 9)))),
                                 "reference": delivery_no, "sku": sku, "description": prod["desc"],
                                 "matdoc_no": "50" + digits(rng, 8), "sales_order": "", "batch": batch,
                                 "plant": DCS[int(rng.integers(0, 4))], "product_division": prod["div"], "quantity": n})

    for kind, count in (("ontime", 140), ("late", 70), ("nodel", 40), ("nc_received", 50), ("nc_pending", 20)):
        for _ in range(count):
            add_history(kind)

    # ---- noise ----------------------------------------------------------------------------------------
    for _ in range(400):
        plant = (DCS + FIELD_PLANTS)[int(rng.integers(0, 7))]
        prod = products[int(rng.integers(0, len(products)))]
        d = today - timedelta(days=int(rng.integers(5, 900)))
        movement.append({"movement_type": [WHR, RET][int(rng.integers(0, 2))], "posting_date": d, "reference": "08" + digits(rng, 8),
                         "sku": prod["sku"], "description": prod["desc"], "matdoc_no": "49" + digits(rng, 8), "sales_order": "",
                         "batch": "N" + digits(rng, 7), "plant": plant, "product_division": prod["div"], "quantity": int(rng.integers(1, 5))})
    for _ in range(260):
        prod = products[int(rng.integers(0, len(products)))]
        c = customers[int(rng.integers(0, len(customers)))]
        plant = (FIELD_PLANTS + CONS_PLANTS)[int(rng.integers(0, 5))]
        q = int(rng.integers(1, 6))
        cockpit.append({"branch": f"{plant}_{int(rng.integers(1, 9)):02d}", "customer_no": c["no"], "sku": prod["sku"].zfill(18),
                        "description": prod["desc"], "batch": "N" + digits(rng, 7), "vendor_batch": "", "serial": digits(rng, 9),
                        "expiry": today + timedelta(days=int(rng.integers(-30, 700))), "customer_name": c["name"],
                        "qty_on_hand": q, "qty_restricted": 0, "qty_in_transit": 0, "qty_total": q})
    for _ in range(140):
        so = "5" + digits(rng, 9)
        carrier_rows(so, "PO" + digits(rng, 7), [digits(rng, 12)], ["label", "moving", "delivered"][int(rng.integers(0, 3))],
                     today - timedelta(days=int(rng.integers(10, 250))))

    master = [{"customer_no": c["no"], "customer_name": c["name"], "sold_to_no": c["sold_to"], "sold_to_name": c["name"],
               "sold_to_alt": c["sold_to"], "email": c["email"], "address1": c["addr"], "city": c["city"], "state": c["state"],
               "zip": c["zip"], "phone": c["phone"]} for c in customers + field_accounts["A"] + field_accounts["B"]]

    return {
        "delivery": pd.DataFrame(delivery), "movement": pd.DataFrame(movement), "cockpit": pd.DataFrame(cockpit),
        "customer_master": pd.DataFrame(master), "carrier_status": pd.DataFrame(carrier),
        "owner_alignment": alignment, "expected_actions": pd.DataFrame(expected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--today", default="2026-10-23", help="anchor date for every generated date (YYYY-MM-DD)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = build(args.seed, date.fromisoformat(args.today))
    for name, df in tables.items():
        target = out / f"{name}.csv"
        df.to_csv(target, index=False)
        print(f"{target.name}: {len(df):,} rows")
    exp = tables["expected_actions"]
    print("expected (owner, group) counts:")
    print(exp.groupby(["expected_owner", "expected_group"]).size().to_string())


if __name__ == "__main__":
    main()
