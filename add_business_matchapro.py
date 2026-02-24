import requests
import time
import base64
import pandas as pd
from tqdm import tqdm
from scraping_multithread.login import login_with_sso

# ------------------------------------------------------
# KONFIGURASI - HARUS DIUPDATE SESUAI STATUS TERBARU
# ------------------------------------------------------
BASE_URL = "https://matchapro.web.bps.go.id/dirgc/draft-tambah-usaha"

HEADERS = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Origin': 'https://matchapro.web.bps.go.id',
    'Referer': 'https://matchapro.web.bps.go.id/dirgc',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 12; M2010J19CG Build/SKQ1.211202.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/143.0.7499.192 Mobile Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': 'Android WebView',
    'sec-ch-ua-mobile': '1',
    'sec-ch-ua-platform': 'Android',
    # 'Cookie': 'f5avraaaaaaaaaaaaaaaa_session_=HGFALLMCPCJCFPKOFMGNBIFILKCCKCJADODLMLFOPLNIGAKGDELMMHECHDEHPHALIMNDOCFBGCPHBJNOPIMAFBBDGCKBDHNKEADKEHLCJBEIGNOLGHDFMMCLNHAMIEPF; _ga_K98R6MSKRH=GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0; _ga=GA1.3.1178577724.1767583475; _ga_XXTTVXWHDB=GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0; TS014fcb37=0167a1c8615e42ab1b58e70e9223288d4d45e334ca539806e1c054fc1da49c634c048daee2ee67f6443b937b7be37dc109857ea826; BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; TS0151fc2b=0167a1c861a19ddee811a5a194c807d5303a9362b38cfb29bc9e0365585ed1289f95139c256f3e51b635860ba71c466070235f224c; f5avraaaaaaaaaaaaaaaa_session_=FFCJAEHCIABMCNAPMAAGDFMKALBDGHEPACHNFIOEFINJGHPPHBBNJGGKFFBFFFNAPPIDDJNDOCIGPJFJNPJAEOBIGCAJIAPAKGAOLODDMKKLNDGGFNMPBFGFOGNIKMEP; XSRF-TOKEN=eyJpdiI6IlMwZ05wZGplSzZNdzZoc0hYOGtYWFE9PSIsInZhbHVlIjoidWplWFdtSkNOYlMxRjNTWTRUQnpiUXFvdVh1S3ZJTFhFY21pQW40Y3R5K0g5dHlPMVpMc1dpRjliQ2FRSmNQajlPMHI3bFFCQ2gwT0xFdHJLRjVoSVEyK2lidW51bjdRWjlTcE9UbUlZQ0YzUFU1Z2NSbjZXRXRZbGxXaDg2VGUiLCJtYWMiOiJjMTNkYzQzOTEzYjlmOTIzMGIyMmI0ZjcxZWRlMzk5NjQyYTc5NThiYTgzNTVjYjNlZjQxNzFmOGZiOGE0M2M3IiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6IjZzNFZuTVVjWDVhUy82VW5sTzQvVkE9PSIsInZhbHVlIjoiT1FoQ3U4YVU0MWZLVkd1K0dqRDNyZ0s4MTJkYWtvZlhIV1QxQ2NIRjNGdnp3ZlprSDNEZ2NudG55eVRhaW04bXFyZHVZTlQxVlBxMC9FQTNJakdpU09FWC9sRGlURi94NmRNdG4vV2habW5ZcjBaRk9NelRKaTVMV0pJalkramQiLCJtYWMiOiJkYmM1ZDEzMjg1ZjNjY2YxMDUyMTllMThkOGYxOTljYTI5YWFiZWFhNTU2MGYwNmJiOTRkYjgyY2ZmNGY4ZTNhIiwidGFnIjoiIn0%3D; TS43cd8bce027=0815dd1fcdab2000ebb0eaf2bd2dde3b8afe0148105513540c97d4cf1eb9fea315b31db36133f1cc0804b4267e1130009904862427f925ef017d8cd841b068fc58fb04aa485af90d044a5c0e6c4f4e880456029b4a2857c92646cbd2ff2e1d31',
}


# Payload dasar
BASE_PAYLOAD = data = {
    '_token': '',
    'id_table': '',
    'nama_usaha': '',
    'alamat': '',
    'provinsi': '',
    'kabupaten': '',
    'kecamatan': '',
    'desa': '',
    'latitude': '',
    'longitude': '',
    'confirmSubmit': 'false',
    'totalSimilar': '0',
}

# Sumber file input
SOURCE_KDM_NOT_MATCHED = "result/kdm_not_matched.csv"
AREA_KABUPATEN_FILE = "result/area_level_kabupaten.csv"
AREA_KECAMATAN_FILE = "result/area_level_kecamatan.csv"
AREA_DESA_FILE = "result/area_level_desa.csv"
PROVINSI_FIXED = "120"

DELAY_BETWEEN_REQUEST = 1.3     # detik, jangan terlalu kecil
MAX_RETRY_PER_ROW = 3
DEBUG_NUMBER = 1  # 0 = proses semua baris, >0 = batasi jumlah baris untuk testing
PRINT_RESPONSE_RESULT = True
PRINT_PAYLOAD_RESULT = True
# ------------------------------------------------------

def normalize_cell_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def b64_encode_unicode(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def build_nama_usaha(name_value, owner_value):
    name = normalize_cell_value(name_value)
    owner = normalize_cell_value(owner_value)
    if owner:
        return f"{name} <{owner}>" if name else f"<{owner}>"
    return name


def load_area_maps():
    df_kab = pd.read_csv(AREA_KABUPATEN_FILE, dtype=str, low_memory=False).fillna("")
    df_kec = pd.read_csv(AREA_KECAMATAN_FILE, dtype=str, low_memory=False).fillna("")
    df_desa = pd.read_csv(AREA_DESA_FILE, dtype=str, low_memory=False).fillna("")

    kab_map = {}
    for _, row in df_kab.iterrows():
        kab_kode = normalize_cell_value(row.get("kode", "")).zfill(2)
        if not kab_kode:
            continue
        key = f"35{kab_kode}"
        kab_map[key] = normalize_cell_value(row.get("id", ""))

    kec_map = {}
    for _, row in df_kec.iterrows():
        kab_kode = normalize_cell_value(row.get("kabupaten_kode", "")).zfill(2)
        kec_kode = normalize_cell_value(row.get("kode", "")).zfill(3)
        if not kab_kode or not kec_kode:
            continue
        key = f"35{kab_kode}{kec_kode}"
        kec_map[key] = normalize_cell_value(row.get("id", ""))

    desa_map = {}
    for _, row in df_desa.iterrows():
        kab_kode = normalize_cell_value(row.get("kabupaten_kode", "")).zfill(2)
        kec_kode = normalize_cell_value(row.get("kecamatan_kode", "")).zfill(3)
        desa_kode = normalize_cell_value(row.get("kode", "")).zfill(3)
        if not kab_kode or not kec_kode or not desa_kode:
            continue
        key = f"35{kab_kode}{kec_kode}{desa_kode}"
        desa_map[key] = normalize_cell_value(row.get("id", ""))

    return kab_map, kec_map, desa_map


def build_payload_from_kdm_row(row, kab_map, kec_map, desa_map):
    sls_id = normalize_cell_value(row.get("sls_id", ""))
    sls4 = sls_id[:4]
    sls7 = sls_id[:7]
    sls10 = sls_id[:10]
    nama_usaha_raw = build_nama_usaha(row.get("name", ""), row.get("owner", ""))
    alamat_raw = "test"

    payload = BASE_PAYLOAD.copy()
    payload["nama_usaha"] = b64_encode_unicode(nama_usaha_raw)
    payload["alamat"] = b64_encode_unicode(alamat_raw)
    payload["provinsi"] = PROVINSI_FIXED
    payload["kabupaten"] = kab_map.get(sls4, "")
    payload["kecamatan"] = kec_map.get(sls7, "")
    payload["desa"] = desa_map.get(sls10, "")
    payload["latitude"] = normalize_cell_value(row.get("latitude", ""))
    payload["longitude"] = normalize_cell_value(row.get("longitude", ""))
    return payload


def post_with_retry(payload, cookies_dict, row_number):
    for attempt in range(1, MAX_RETRY_PER_ROW + 1):
        try:
            response = requests.post(
                BASE_URL,
                cookies=cookies_dict,
                headers=HEADERS,
                data=payload,
                timeout=20,
            )
            response.raise_for_status()
            return True, response.text
        except Exception as e:
            if attempt == MAX_RETRY_PER_ROW:
                return False, str(e)
            print(
                f"Baris {row_number}: percobaan ke-{attempt} gagal ({e}). "
                f"Ulangi dalam 3 detik..."
            )
            time.sleep(3)


def main():
    print("Melakukan login otomatis...\n")

    # Login credentials - update as needed
    username = input("Masukkan username: ")
    password = input("Masukkan password: ")
    otp_code = input("Masukkan OTP (kosongkan jika tidak ada): ").strip() or None

    # Perform login
    page, browser = login_with_sso(username, password, otp_code)

    if not page:
        print("Login gagal. Tidak dapat melanjutkan scraping.")
        return

    try:
        # Navigate to /dirgc to get _token
        url_gc = "https://matchapro.web.bps.go.id/dirgc"
        page.goto(url_gc)
        page.wait_for_load_state('networkidle')

        # Wait for CSRF token meta tag to be attached
        page.wait_for_selector('meta[name="csrf-token"]', state='attached', timeout=10000)

        # Extract _token
        token_element = page.locator('meta[name="csrf-token"]')
        if token_element.count() > 0:
            _token = token_element.get_attribute('content')
            BASE_PAYLOAD["_token"] = _token
            print(f"_token diperoleh: {_token}")
        else:
            print("Gagal mendapatkan _token")
            browser.close()
            return

        # Get cookies
        cookies = page.context.cookies()
        cookie_string = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
        HEADERS["cookie"] = cookie_string
        print("Cookies diperoleh dan diset ke headers")

    except Exception as e:
        print(f"Error saat login atau ekstraksi: {e}")
        browser.close()
        return

    # Close browser after getting credentials
    browser.close()

    print("Login berhasil. Memulai insert data dari kdm_not_matched...\n")

    try:
        kab_map, kec_map, desa_map = load_area_maps()
    except Exception as e:
        print(f"Gagal membaca file area level: {e}")
        return

    try:
        df_source = pd.read_csv(SOURCE_KDM_NOT_MATCHED, dtype=str, low_memory=False)
    except Exception as e:
        print(f"Gagal membaca file sumber ({SOURCE_KDM_NOT_MATCHED}): {e}")
        return

    if df_source.empty:
        print("File sumber kosong. Tidak ada data untuk diinsert.")
        return

    if DEBUG_NUMBER and DEBUG_NUMBER > 0:
        df_source = df_source.head(DEBUG_NUMBER).copy()
        print(f"DEBUG aktif: hanya memproses {len(df_source)} baris pertama")

    total_rows = len(df_source)
    print(f"Total baris pada sumber: {total_rows:,}")
    print(f"Kolom sumber: {', '.join(df_source.columns.astype(str).tolist())}\n")

    cookies_dict = {cookie['name']: cookie['value'] for cookie in cookies}
    success_count = 0
    failed_rows = []
    skipped_rows = []

    with tqdm(total=total_rows, desc="Insert Progress", unit="row") as pbar:
        for idx, row in df_source.iterrows():
            row_number = idx + 2  # +2 karena header csv di baris 1
            payload = build_payload_from_kdm_row(row, kab_map, kec_map, desa_map)

            if not payload["nama_usaha"]:
                skipped_rows.append({"row": row_number, "reason": "nama_usaha kosong"})
                pbar.update(1)
                continue
            if not payload["kabupaten"] or not payload["kecamatan"] or not payload["desa"]:
                skipped_rows.append({
                    "row": row_number,
                    "reason": "mapping area tidak lengkap",
                    "sls_id": normalize_cell_value(row.get("sls_id", "")),
                })
                pbar.update(1)
                continue

            if PRINT_PAYLOAD_RESULT:
                print(f"[PAYLOAD] Baris {row_number}: {payload}")

            ok, result = post_with_retry(payload, cookies_dict, row_number)
            if ok:
                success_count += 1
                if PRINT_RESPONSE_RESULT:
                    print(f"[OK] Baris {row_number} response: {result}")
            else:
                failed_rows.append({"row": row_number, "error": result})
                if PRINT_RESPONSE_RESULT:
                    print(f"[FAILED] Baris {row_number} response/error: {result}")

            pbar.update(1)
            time.sleep(DELAY_BETWEEN_REQUEST)

    print(f"\nSelesai proses insert.")
    print(f"Berhasil : {success_count}")
    print(f"Gagal    : {len(failed_rows)}")
    print(f"Skip     : {len(skipped_rows)}")

    if failed_rows:
        print("\nDetail baris gagal:")
        for item in failed_rows[:20]:
            print(f"- Baris {item['row']}: {item['error']}")
        if len(failed_rows) > 20:
            print(f"... dan {len(failed_rows) - 20} baris gagal lainnya")

    if skipped_rows:
        print("\nDetail baris skip:")
        for item in skipped_rows[:20]:
            if "sls_id" in item:
                print(f"- Baris {item['row']}: {item['reason']} (sls_id={item['sls_id']})")
            else:
                print(f"- Baris {item['row']}: {item['reason']}")
        if len(skipped_rows) > 20:
            print(f"... dan {len(skipped_rows) - 20} baris skip lainnya")


if __name__ == "__main__":
    main()
