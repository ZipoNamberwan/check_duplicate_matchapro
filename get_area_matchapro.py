import requests
import time
import os
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
    'provinsi': '120',
}

DELAY_BETWEEN_REQUEST = 0.2     # detik, jangan terlalu kecil
MAX_RETRY_PER_ROW = 3
RESULT_FOLDER = "result"
OUTPUT_KABUPATEN = "area_level_kabupaten.csv"
OUTPUT_KECAMATAN = "area_level_kecamatan.csv"
OUTPUT_DESA = "area_level_desa.csv"
# ------------------------------------------------------

def normalize_cell_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_payload_from_row(row):
    payload = BASE_PAYLOAD.copy()
    for key in payload.keys():
        if key in row:
            payload[key] = normalize_cell_value(row[key])
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

    print("Login berhasil. Mengambil area (kabupaten/kota user)...\n")

    cookies_dict = {cookie['name']: cookie['value'] for cookie in cookies}
    data = BASE_PAYLOAD.copy()

    try:
        response = requests.post(
            'https://matchapro.web.bps.go.id/wil-kabupaten-kota-user',
            cookies=cookies_dict,
            headers=HEADERS,
            data=data,
            timeout=20,
        )
        response.raise_for_status()

        print(f"Status code: {response.status_code}")
        try:
            print("Response JSON:")
            kabupaten_list = response.json()
            print(kabupaten_list)

            if not isinstance(kabupaten_list, list):
                print("Format response kabupaten tidak berupa list, skip request wil-kecamatan.")
                return

            print("\nMengambil data kecamatan untuk setiap kabupaten/kota...")
            total_kecamatan = 0
            total_desa = 0
            kabupaten_records = []
            kecamatan_records = []
            desa_records = []

            for kab_item in kabupaten_list:
                kabupaten_records.append({
                    "id": str(kab_item.get("id", "")).strip(),
                    "kode": str(kab_item.get("kode", "")).strip(),
                    "nama": str(kab_item.get("nama", "")).strip(),
                    "provinsi_id": str(kab_item.get("provinsi_id", "")).strip(),
                    "locked": str(kab_item.get("locked", "")).strip(),
                })

            for item in tqdm(kabupaten_list, desc="Progress kecamatan", unit="kabupaten"):
                kab_id = str(item.get("id", "")).strip()
                kab_nama = str(item.get("nama", "")).strip()
                kab_kode = str(item.get("kode", "")).strip()
                kab_provinsi_id = str(item.get("provinsi_id", "")).strip()
                kab_locked = str(item.get("locked", "")).strip()
                if not kab_id:
                    continue

                kec_payload = {
                    'kabupaten_kota': kab_id,
                    '_token': BASE_PAYLOAD.get("_token", ""),
                }

                kec_response = None
                last_error = None
                for attempt in range(1, MAX_RETRY_PER_ROW + 1):
                    try:
                        kec_response = requests.post(
                            'https://matchapro.web.bps.go.id/wil-kecamatan',
                            cookies=cookies_dict,
                            headers=HEADERS,
                            data=kec_payload,
                            timeout=20,
                        )
                        kec_response.raise_for_status()
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < MAX_RETRY_PER_ROW:
                            time.sleep(3)

                if kec_response is None:
                    print(f"Gagal wil-kecamatan untuk {kab_nama} ({kab_id}): {last_error}")
                    continue

                try:
                    kec_json = kec_response.json()
                    jumlah_kec = len(kec_json) if isinstance(kec_json, list) else 0
                    total_kecamatan += jumlah_kec
                    print(f"{kab_nama} ({kab_id}) -> {jumlah_kec} kecamatan")

                    if isinstance(kec_json, list):
                        for kec_item in kec_json:
                            kec_id = str(kec_item.get("id", "")).strip()
                            kec_nama = str(kec_item.get("nama", "")).strip()

                            kecamatan_records.append({
                                "id": kec_id,
                                "kode": str(kec_item.get("kode", "")).strip(),
                                "nama": kec_nama,
                                "kabupaten_kota": str(kec_item.get("kabupaten_kota", "")).strip(),
                                "kabupaten_id": kab_id,
                                "kabupaten_kode": kab_kode,
                                "kabupaten_nama": kab_nama,
                                "kabupaten_provinsi_id": kab_provinsi_id,
                                "kabupaten_locked": kab_locked,
                                "locked": str(kec_item.get("locked", "")).strip(),
                            })

                            if not kec_id:
                                continue

                            desa_payload = {
                                'kecamatan': kec_id,
                                '_token': BASE_PAYLOAD.get("_token", ""),
                            }

                            desa_response = None
                            desa_last_error = None
                            for desa_attempt in range(1, MAX_RETRY_PER_ROW + 1):
                                try:
                                    desa_response = requests.post(
                                        'https://matchapro.web.bps.go.id/wil-desa',
                                        cookies=cookies_dict,
                                        headers=HEADERS,
                                        data=desa_payload,
                                        timeout=20,
                                    )
                                    desa_response.raise_for_status()
                                    break
                                except Exception as desa_error:
                                    desa_last_error = desa_error
                                    if desa_attempt < MAX_RETRY_PER_ROW:
                                        time.sleep(3)

                            if desa_response is None:
                                print(f"Gagal wil-desa untuk {kec_nama} ({kec_id}): {desa_last_error}")
                                continue

                            try:
                                desa_json = desa_response.json()
                                if isinstance(desa_json, list):
                                    total_desa += len(desa_json)
                                    for desa_item in desa_json:
                                        desa_records.append({
                                            "id": str(desa_item.get("id", "")).strip(),
                                            "kode": str(desa_item.get("kode", "")).strip(),
                                            "nama": str(desa_item.get("nama", "")).strip(),
                                            "kecamatan": str(desa_item.get("kecamatan", "")).strip(),
                                            "kecamatan_id": kec_id,
                                            "kecamatan_kode": str(kec_item.get("kode", "")).strip(),
                                            "kecamatan_nama": kec_nama,
                                            "kecamatan_kabupaten_kota": str(kec_item.get("kabupaten_kota", "")).strip(),
                                            "kabupaten_id": kab_id,
                                            "kabupaten_kode": kab_kode,
                                            "kabupaten_nama": kab_nama,
                                            "kabupaten_provinsi_id": kab_provinsi_id,
                                            "kabupaten_locked": kab_locked,
                                            "locked": str(desa_item.get("locked", "")).strip(),
                                        })
                            except Exception:
                                print(f"{kec_nama} ({kec_id}) -> response wil-desa non-JSON")

                            time.sleep(DELAY_BETWEEN_REQUEST)
                except Exception:
                    print(f"{kab_nama} ({kab_id}) -> response non-JSON")

                time.sleep(DELAY_BETWEEN_REQUEST)

            print(f"\nSelesai. Total kecamatan terambil: {total_kecamatan}")
            print(f"Selesai. Total desa terambil: {total_desa}")

            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)

            kab_path = os.path.join(output_dir, OUTPUT_KABUPATEN)
            kec_path = os.path.join(output_dir, OUTPUT_KECAMATAN)
            desa_path = os.path.join(output_dir, OUTPUT_DESA)

            pd.DataFrame(kabupaten_records).to_csv(kab_path, index=False, encoding="utf-8")
            pd.DataFrame(kecamatan_records).to_csv(kec_path, index=False, encoding="utf-8")
            pd.DataFrame(desa_records).to_csv(desa_path, index=False, encoding="utf-8")

            print("\nBerhasil menyimpan area 3 level:")
            print(f"- Kabupaten: {kab_path} ({len(kabupaten_records)} rows)")
            print(f"- Kecamatan: {kec_path} ({len(kecamatan_records)} rows)")
            print(f"- Desa: {desa_path} ({len(desa_records)} rows)")
        except Exception:
            print("Response text:")
            print(response.text)
    except Exception as e:
        print(f"Gagal mengambil area: {e}")


if __name__ == "__main__":
    main()