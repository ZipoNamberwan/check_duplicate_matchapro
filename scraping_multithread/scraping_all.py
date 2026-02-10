import requests
import time
import pandas as pd
from tqdm import tqdm
from login import login_with_sso
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os

# ------------------------------------------------------
# KONFIGURASI - HARUS DIUPDATE SESUAI STATUS TERBARU
# ------------------------------------------------------
BASE_URL = "https://matchapro.web.bps.go.id/direktori-usaha/data-gc-card"

HEADERS = {
    "host": "matchapro.web.bps.go.id",
    "connection": "keep-alive",
    "sec-ch-ua": "\"Android WebView\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": "\"Android\"",
    "x-requested-with": "com.matchapro.app",
    "user-agent": "Mozilla/5.0 (Linux; Android 12; M2010J19CG Build/SKQ1.211202.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/143.0.7499.192 Mobile Safari/537.36",
    "accept": "*/*",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "origin": "https://matchapro.web.bps.go.id",
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
    "referer": "https://matchapro.web.bps.go.id/dirgc",
    "accept-language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    # !!! WAJIB GANTI !!!
    "cookie": "BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; TS0151fc2b=0167a1c861a692c4c516ea14cef77f079192991528078a048c37cd54af05dc68e3bbdd55effe6a9d9a6447ad6a4cbbf0f206c773d6; XSRF-TOKEN=eyJpdiI6IjFRUGp1N2lTMVVCcW5kWmg0dGozMEE9PSIsInZhbHVlIjoiOWlrM2ZYSmI4UWV6YzMwd1dKdENPdzgrTVZISkNSSzJtMi9OUWplMDRpZlo3UTBIZUkwczNtQTlnNXJQUkpnK3JHajJ5TGJUTnVTaVY2UVJia2FkUFZzVWl2Z0I2aGpwWGZ2a2xWRE8xelRkaEx6dGRuaEphaGVOQWdoR25uQlQiLCJtYWMiOiIwYzVhODNlZGJjYWZjNzQxMjljNDYyNzEwYjdjOGI2Zjk0NTQxMGRkNWJhYTc3NmEwZGE5ZmJlM2U5OGEwMDhkIiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6IkI5SHN4My9WY3Nkc3hXdHdzVkN1R2c9PSIsInZhbHVlIjoiYnNUcUd6Ynd2ZHdXRGxyN1E3STZKaFBYczd3RlI5ZE9JbFdBZ0tTUnRwK29STnFtMXVDMkpLVW1CNXg1M29mQWloWlRDNEt2T0xKdHYzTzllUEt6WmRCSVlsNEliTWgxNks4eVpBUm1jTnJ0d3VyeGNFSnRrVmFYMDhMelVOVHUiLCJtYWMiOiI2MmJkMmYyN2JhNWJkMGIwMjIyOGVhMzUyZWY4YjcxMWQ5OTc4MTYwZTM0OTIxOWYyZmI4ZGQwOTRlZWE2NTNhIiwidGFnIjoiIn0%3D; TS1a53eee7027=0815dd1fcdab2000ead5618125788d97a7b8eecb3c910422f443944af41075f98354ebfab66871a8088d9fcc1e1130009eb4655c9019dcd2648c18388ab84ebf5f44df9d38e4035a68a6c8481af5e0a6ff2c2b8df2dec2198abeb80fff0b5242"
}

# Payload dasar
BASE_PAYLOAD = {
    "_token": "", # Will be set automatically
    "start": 0,
    "length": 1000, # Kurangi dari 2000 untuk menghindari response terpotong
    "nama_usaha": "",
    "alamat_usaha": "",
    "provinsi": "", # akan diisi dari parsing cari_kode.htm
    "kabupaten": "", # akan diisi dari parsing cari_kode.htm
    "kecamatan": "",
    "desa": "",
    "status_filter": "semua",
    "rtotal": "0",
    "sumber_data": "",
    "skala_usaha": "",
    "idsbr": "",
    "history_profiling": ""
}

# Nama file output
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "source_matcha_pro_all")
OUTPUT_CSV_FALLBACK = os.path.join(OUTPUT_FOLDER, "combined_data.csv")

DELAY_BETWEEN_REQUEST = 1.3     # detik, jangan terlalu kecil
MAX_WORKERS = 5  # Jumlah thread concurrent, sesuaikan dengan kemampuan server

# Lock untuk thread-safe printing
print_lock = Lock()

# Global variables untuk provinsi dan kabupaten list
KABUPATEN_LIST = {
    # "2389": "[01] PACITAN",
    # "2390": "[02] PONOROGO",
    # "2391": "[03] TRENGGALEK",
    # "2392": "[04] TULUNGAGUNG",
    # "2393": "[05] BLITAR",
    # "2394": "[06] KEDIRI",
    # "2395": "[07] MALANG",
    # "2396": "[08] LUMAJANG",
    "2397": "[09] JEMBER",
    # "2398": "[10] BANYUWANGI",
    # "2399": "[11] BONDOWOSO",
    # "2400": "[12] SITUBONDO",
    # "2401": "[13] PROBOLINGGO",
    # "2402": "[14] PASURUAN",
    "2403": "[15] SIDOARJO",
    # "2404": "[16] MOJOKERTO",
    # "2405": "[17] JOMBANG",
    # "2406": "[18] NGANJUK",
    # "2407": "[19] MADIUN",
    # "2408": "[20] MAGETAN",
    # "2409": "[21] NGAWI",
    # "2410": "[22] BOJONEGORO",
    # "2411": "[23] TUBAN",
    # "2412": "[24] LAMONGAN",
    "2413": "[25] GRESIK",
    # "2414": "[26] BANGKALAN",
    # "2415": "[27] SAMPANG",
    "2416": "[28] PAMEKASAN",
    # "2417": "[29] SUMENEP",
    # "2418": "[71] KEDIRI",
    # "2419": "[72] BLITAR",
    # "2420": "[73] MALANG",
    # "2421": "[74] PROBOLINGGO",
    # "2422": "[75] PASURUAN",
    # "2423": "[76] MOJOKERTO",
    # "2424": "[77] MADIUN",
    # "2425": "[78] SURABAYA",
    # "2426": "[79] BATU",
}
KODE_PROVINSI = 120
# ------------------------------------------------------


def fetch_page(start, length, kode_kabupaten=""):
    payload = BASE_PAYLOAD.copy()
    payload["start"] = str(start)
    payload["length"] = str(length)
    payload["kabupaten"] = kode_kabupaten  # Set kabupaten spesifik

    try:
        r = requests.post(BASE_URL, data=payload, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        with print_lock:
            print(f"Error saat mengambil data (start={start}, kab={kode_kabupaten}): {e}")
        return None


def get_kabupaten_list(html_content):
    """Extract list of kabupaten from HTML select options"""
    kabupaten_list = []
    
    # Find the kabupaten select element and extract all options
    kab_select_match = re.search(r'<select[^>]*id="f_kabupaten"[^>]*>(.*?)</select>', html_content, re.DOTALL)
    
    if kab_select_match:
        select_content = kab_select_match.group(1)
        # Find all option tags with value and text
        options = re.findall(r'<option value="(\d+)"[^>]*>(.*?)</option>', select_content)
        
        for kode, nama in options:
            if kode and kode != "":  # Skip empty values
                kabupaten_list.append({
                    'kode': kode,
                    'nama': nama.strip()
                })
    
    return kabupaten_list


def normalize_kabupaten_list(items):
    """Normalize kabupaten list into list of dicts with kode/nama."""
    if not items:
        return []
    if isinstance(items, dict):
        return [{"kode": str(k).strip(), "nama": str(v).strip()} for k, v in items.items() if str(k).strip()]
    if isinstance(items[0], dict):
        return items
    normalized = []
    for item in items:
        kode = str(item).strip()
        if kode:
            normalized.append({"kode": kode, "nama": f"Kabupaten {kode}"})
    return normalized


def process_kabupaten(kode_kab, nama_kab):
    """Process all data for one kabupaten"""
    global KODE_PROVINSI
    
    with print_lock:
        print(f"\n{'='*70}")
        print(f"Memproses Kabupaten: {nama_kab} (Kode: {kode_kab})")
        print(f"{'='*70}")
    
    # Cek total data untuk kabupaten ini
    first_response = fetch_page(0, 100, kode_kab)
    if not first_response or "recordsTotal" not in first_response:
        with print_lock:
            print(f"[{nama_kab}] Gagal mendapatkan informasi awal")
        return None
    
    total_records = first_response["recordsTotal"]
    
    with print_lock:
        print(f"[{nama_kab}] Total data: {total_records:,} record")
    
    if total_records == 0:
        with print_lock:
            print(f"[{nama_kab}] Tidak ada data, skip")
        return None
    
    all_records = []
    length_per_request = 1000
    
    # Progress bar per kabupaten
    with tqdm(total=total_records, desc=f"{nama_kab[:20]}", unit="rec", position=None, leave=True) as pbar:
        start = 0
        retry_count = 0
        max_retries = 3
        
        while start < total_records:
            data = fetch_page(start, length_per_request, kode_kab)
            
            if not data or "data" not in data or not isinstance(data["data"], list):
                retry_count += 1
                if retry_count >= max_retries:
                    with print_lock:
                        print(f"\n[{nama_kab}] Gagal setelah {max_retries} percobaan di start={start}")
                    break
                
                with print_lock:
                    print(f"\n[{nama_kab}] Gagal di start={start}, retry {retry_count}/{max_retries}...")
                time.sleep(6)
                continue
            
            page_data = data["data"]
            all_records.extend(page_data)
            
            fetched_this_time = len(page_data)
            pbar.update(fetched_this_time)
            
            start += fetched_this_time
            retry_count = 0  # Reset retry count on success
            
            time.sleep(DELAY_BETWEEN_REQUEST)
    
    if not all_records:
        with print_lock:
            print(f"[{nama_kab}] Tidak ada data berhasil dikumpulkan")
        return None
    
    # Bersihkan data
    for record in all_records:
        if 'alamat_usaha' in record and isinstance(record['alamat_usaha'], str):
            record['alamat_usaha'] = record['alamat_usaha'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        if 'kegiatan_usaha' in record and isinstance(record['kegiatan_usaha'], str):
            record['kegiatan_usaha'] = record['kegiatan_usaha'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        if 'nama_usaha' in record and isinstance(record['nama_usaha'], str):
            record['nama_usaha'] = record['nama_usaha'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    
    # Simpan per kabupaten
    df = pd.DataFrame(all_records)
    
    # Create output folder if not exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Sanitize filename
    safe_nama = re.sub(r'[<>:"/\\|?*]', '_', nama_kab)
    output_file = os.path.join(OUTPUT_FOLDER, f"kabupaten_{kode_kab}_{safe_nama}.csv")
    
    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        with print_lock:
            print(f"[{nama_kab}] Berhasil disimpan: {output_file} ({len(all_records):,} record)")
    except Exception as e:
        with print_lock:
            print(f"[{nama_kab}] Gagal menyimpan: {e}")
        return None
    
    return {
        'kode': kode_kab,
        'nama': nama_kab,
        'records': len(all_records),
        'file': output_file
    }


def main():
    global KABUPATEN_LIST, KODE_PROVINSI
    
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
        # Navigate to /dirgc to get _token and parse HTML for provinsi/kabupaten
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

        # Jika kabupaten sudah diset manual, lewati parsing HTML
        if KABUPATEN_LIST:
            KABUPATEN_LIST = normalize_kabupaten_list(KABUPATEN_LIST)
            BASE_PAYLOAD["provinsi"] = str(KODE_PROVINSI) if KODE_PROVINSI else ""
            print(f"Kode provinsi: {BASE_PAYLOAD['provinsi']}")
            print(f"\nMenggunakan kabupaten manual: {len(KABUPATEN_LIST)}")
            for i, kab in enumerate(KABUPATEN_LIST[:5], 1):
                print(f"  {i}. {kab['nama']} (Kode: {kab['kode']})")
            if len(KABUPATEN_LIST) > 5:
                print(f"  ... dan {len(KABUPATEN_LIST) - 5} kabupaten lainnya")
        else:
            # Parse HTML dari request ke direktori-usaha untuk mendapatkan kode provinsi dan kabupaten
            url_direktori = "https://matchapro.web.bps.go.id/direktori-usaha"
            try:
                response = requests.get(url_direktori, headers=HEADERS, timeout=20)
                response.raise_for_status()
                html_content = response.text
                
                # Cari kode provinsi
                prov_match = re.search(r'<select id="f_provinsi".*?<option value="(\d+)" selected>', html_content, re.DOTALL)
                if prov_match:
                    KODE_PROVINSI = prov_match.group(1)
                else:
                    KODE_PROVINSI = ""  # default
                
                # Update BASE_PAYLOAD provinsi
                BASE_PAYLOAD["provinsi"] = KODE_PROVINSI
                
                print(f"Kode provinsi: {KODE_PROVINSI}")
                
                # Get list of kabupaten
                KABUPATEN_LIST = get_kabupaten_list(html_content)
                
                if not KABUPATEN_LIST:
                    print("Gagal mendapatkan daftar kabupaten")
                    browser.close()
                    return
                
                print(f"\nDitemukan {len(KABUPATEN_LIST)} kabupaten:")
                for i, kab in enumerate(KABUPATEN_LIST[:5], 1):  # Show first 5
                    print(f"  {i}. {kab['nama']} (Kode: {kab['kode']})")
                if len(KABUPATEN_LIST) > 5:
                    print(f"  ... dan {len(KABUPATEN_LIST) - 5} kabupaten lainnya")
            
            except Exception as e:
                print(f"Error saat parsing HTML dari direktori-usaha: {e}")
                browser.close()
                return

    except Exception as e:
        print(f"Error saat login atau ekstraksi: {e}")
        browser.close()
        return

    # Close browser after getting credentials
    browser.close()

    print("\nLogin berhasil. Memulai pengambilan data dengan multithread...\n")
    print(f"Menggunakan {MAX_WORKERS} thread concurrent")
    print(f"Hasil akan disimpan ke folder: {OUTPUT_FOLDER}\n")
    
    # Process kabupaten with multithread
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all kabupaten tasks
        future_to_kab = {
            executor.submit(process_kabupaten, kab['kode'], kab['nama']): kab 
            for kab in KABUPATEN_LIST
        }
        
        # Process completed tasks
        for future in as_completed(future_to_kab):
            kab = future_to_kab[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                with print_lock:
                    print(f"\nError processing {kab['nama']}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("RINGKASAN HASIL")
    print("="*70)
    
    if results:
        total_records = sum(r['records'] for r in results)
        print(f"\nBerhasil download {len(results)} kabupaten")
        print(f"Total record: {total_records:,}")
        print(f"\nDaftar file hasil:")
        for r in sorted(results, key=lambda x: x['nama']):
            print(f"  - {r['nama']}: {r['records']:,} record -> {r['file']}")
        
        # Combine all files into one (auto)
        print("\nMenggabungkan semua file...")
        all_dfs = []
        for r in results:
            try:
                df = pd.read_csv(r['file'])
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {r['file']}: {e}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_file = OUTPUT_CSV_FALLBACK
            os.makedirs(os.path.dirname(combined_file), exist_ok=True)
            combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            print(f"File gabungan disimpan: {combined_file} ({len(combined_df):,} record)")
    else:
        print("\nTidak ada data yang berhasil didownload")
    
    print("\n" + "="*70)
    print("Selesai!")
    print("="*70)


if __name__ == "__main__":
    main()
