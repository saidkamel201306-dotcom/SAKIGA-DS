# -*- coding: utf-8 -*-
"""
Dispatch products from db_sakiga_main.products_main into:
- db_sakiga_men.products_men
- db_sakiga_women.products_women

Uses the SAME text pipeline as training (build_model_text) to avoid train/infer mismatch.
"""
from __future__ import annotations
import joblib
import pymysql
import os
import math

from src.text_utils import build_model_text

# =======================
# CONFIGURATION
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

MYSQL_BASE = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

DB_MAIN = "db_sakiga_main"
DB_MEN = "db_sakiga_men"
DB_WOMEN = "db_sakiga_women"

TABLE_MAIN = "products_main"
TABLE_MEN = "products_men"
TABLE_WOMEN = "products_women"

def mysql_safe(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

print("🗄️ Vérification des bases MEN / WOMEN.")
cnx_admin = pymysql.connect(host="localhost", user="root", password="", charset="utf8mb4")
cur_admin = cnx_admin.cursor()
cur_admin.execute("CREATE DATABASE IF NOT EXISTS db_sakiga_men CHARACTER SET utf8mb4")
cur_admin.execute("CREATE DATABASE IF NOT EXISTS db_sakiga_women CHARACTER SET utf8mb4")
cur_admin.close()
cnx_admin.close()

print("🤖 Chargement du modèle IA.")
model = joblib.load(MODEL_PATH)

cnx_main = pymysql.connect(**MYSQL_BASE, database=DB_MAIN)
cnx_men = pymysql.connect(**MYSQL_BASE, database=DB_MEN)
cnx_women = pymysql.connect(**MYSQL_BASE, database=DB_WOMEN)

cur_main = cnx_main.cursor()
cur_men = cnx_men.cursor()
cur_women = cnx_women.cursor()

create_table_sql = """
CREATE TABLE IF NOT EXISTS {table} (
    id INT PRIMARY KEY,

    masterCategory VARCHAR(100),
    subCategory VARCHAR(100),
    articleType VARCHAR(100),
    baseColour VARCHAR(50),
    season VARCHAR(50),
    year INT,

    `usage` VARCHAR(100),

    productDisplayName VARCHAR(255),
    filename VARCHAR(255),
    link TEXT,

    price DECIMAL(10,2),

    is_promo TINYINT(1),
    promo_percent INT,
    promo_end DATE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

cur_men.execute(create_table_sql.format(table=TABLE_MEN))
cur_women.execute(create_table_sql.format(table=TABLE_WOMEN))

print("📦 Lecture des produits depuis db_sakiga_main.products_main.")
cur_main.execute(f"SELECT * FROM {TABLE_MAIN}")
products = cur_main.fetchall()
print(f"➡️ {len(products)} produits trouvés")

insert_sql = """
INSERT INTO {table}
(
    id,
    masterCategory,
    subCategory,
    articleType,
    baseColour,
    season,
    year,
    `usage`,
    productDisplayName,
    filename,
    link,
    price,
    is_promo,
    promo_percent,
    promo_end
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

processed = 0
for p in products:
    text = build_model_text(
        p.get("productDisplayName"),
        p.get("subCategory"),
        p.get("articleType"),
    )
    prediction = str(model.predict([text])[0]).upper()

    table = TABLE_MEN if prediction == "MEN" else TABLE_WOMEN
    cur = cur_men if prediction == "MEN" else cur_women

    values = (
        p["id"],
        mysql_safe(p.get("masterCategory")),
        mysql_safe(p.get("subCategory")),
        mysql_safe(p.get("articleType")),
        mysql_safe(p.get("baseColour")),
        mysql_safe(p.get("season")),
        p.get("year"),
        mysql_safe(p.get("usage")),
        mysql_safe(p.get("productDisplayName")),
        mysql_safe(p.get("filename")),
        mysql_safe(p.get("link")),
        mysql_safe(p.get("price")),
        mysql_safe(p.get("is_promo")),
        mysql_safe(p.get("promo_percent")),
        mysql_safe(p.get("promo_end")),
    )

    try:
        cur.execute(insert_sql.format(table=table), values)
        processed += 1
    except Exception as e:
        # ignore duplicates, but print other errors
        msg = str(e)
        if "Duplicate entry" not in msg:
            print("❌ Insert error:", e)

cnx_men.commit()
cnx_women.commit()

cur_main.close(); cur_men.close(); cur_women.close()
cnx_main.close(); cnx_men.close(); cnx_women.close()

print(f"✅ Dispatch terminé. {processed} lignes insérées.")
