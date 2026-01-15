import requests
import pandas as pd
import time
from pathlib import Path

# Configuration
EXCEL_FILE = Path('source_gc_update/gc.xlsx')
BASE_URL = 'https://matchapro.web.bps.go.id'


def transform_coordinates(latitude, longitude):
    """
    Transform invalid coordinates to valid ones
    Handles different decimal separators (. or ,) and validates coordinates
    
    Args:
        latitude: The invalid latitude (can be string with . or , as decimal separator)
        longitude: The invalid longitude (can be string with . or , as decimal separator)
        
    Returns:
        Tuple of (valid_latitude, valid_longitude) as strings with proper format
    """
    try:
        # Convert to string and normalize decimal separator (replace , with .)
        lat_str = str(latitude).strip().replace(',', '.')
        lon_str = str(longitude).strip().replace(',', '.')
        
        # Convert to float for validation
        lat_float = float(lat_str)
        lon_float = float(lon_str)
        
        # Validate latitude range (-90 to 90) and longitude range (-180 to 180)
        if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
            # Return as strings with consistent decimal point
            return str(lat_float), str(lon_float)
        else:
            print(f"  ⚠ Coordinates out of valid range: lat={lat_float}, lon={lon_float}")
            return str(lat_float), str(lon_float)
            
    except (ValueError, AttributeError) as e:
        print(f"  ⚠ Error parsing coordinates: lat={latitude}, lon={longitude}, error={str(e)}")
        return str(latitude), str(longitude)


def get_business_id_by_idsbr(idsbr):
    """
    First request: Get business ID (perusahaan_id) by searching with idsbr
    Also extracts geocoding coordinates if available
    
    Args:
        idsbr: The IDSBR value to search for
        
    Returns:
        Dictionary with perusahaan_id, latitude, longitude, latlong_status if found, None otherwise
    """
    cookies = {
        '_ga_K98R6MSKRH': 'GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0',
        '_ga': 'GA1.3.1178577724.1767583475',
        '_ga_XXTTVXWHDB': 'GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0',
        'BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool': '1418199050.20480.0000',
        'TS018af012': '0167a1c86195f9e6113f15cda0f27acd06889f7687265062cb86097f5a4b1cbed6be85a03ab5f8dce26099dc651d5eb4f099d35463e7c794db13e9b3c070e346c6b3d6df85',
        'f5avraaaaaaaaaaaaaaaa_session_': 'PCIDOHHCIABMCNAPMAENNFMKALBDGHEPACHNFIOEFINJGHPPHBBNJGGKFFBFFFNAPPIDDJNDKHNGLDAJNPJAEOBIPIPJBKAAKGAOLODDMKKLNDLGFNMPBFGFOGNIKMHN',
        'XSRF-TOKEN': 'eyJpdiI6Ik84bkltTlZldFdkbXBhRlhKWmNMUkE9PSIsInZhbHVlIjoiaXJiQzl4QzYxdDVNYjMzcXgzL1B1WHJjZVkrVDkwdGE1WmdPOG1EK3dsSFk4NlZmenF6WHdxOWRXTjJXRFlWMjduSmptaEk5RUhuem44bFlOYWhtTE12dXV2T1FrTFhUV3loNGFvN01VSlZYMUlDNDUrY2w1OStyc25QcFJMRVkiLCJtYWMiOiJkYTFlY2VkMzlmNjU1NzUwNDNmNGQyMzIwM2E5MzcyNjk1OTA0ZWQxMGM5ODFhMTI4NzIxNjlkZGQwN2FlMjE5IiwidGFnIjoiIn0%3D',
        'laravel_session': 'eyJpdiI6IlArbUUxNnF0RjJnR1BQQ0NnVkhVVGc9PSIsInZhbHVlIjoiMDlnMkI3NTk0ak9LekZWNlllYjhuelFvdDNjWnJUUW8wSWJ4bWhRbFR5STBqcHhEdCsybTZpNFlGaFM4Um0wWG5vN3hMcWVBK21rVkZGUFBrWEVxWjhoQ1ZkNzk2NkhlOXBpL3N4a1YyeHQ4VkV3OE14aTBkU1g3SWs3cFJWREciLCJtYWMiOiI1Nzg5ZGFmYjRmYzExZmRhMWVlNDU2NzcxMDdjMjgxYjcwMjY3NDJlODdlMjAxNTI4YTAwZWRiMWI5Zjg1YzYzIiwidGFnIjoiIn0%3D',
        'TS0151fc2b': '0167a1c86118d45785fd31603fab161c6619d4bedf2466c608ef78ed187fcca638455497665181a619bd1ce01ea165dee053fb76bf',
        'TS1a53eee7027': '0815dd1fcdab20008a2279bea573cf45d96709f9410736905970d09471e5359d58e67dc53e23e80508a5506749113000a277bdcccdddff1aa85dea369ddc0839e951d8f0a4152604d03bcb32585e4df0093c0c39fa629154a06431f2aa11386d',
    }

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'https://matchapro.web.bps.go.id',
        'Referer': 'https://matchapro.web.bps.go.id/dirgc',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        # 'Cookie': '_ga_K98R6MSKRH=GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0; _ga=GA1.3.1178577724.1767583475; _ga_XXTTVXWHDB=GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0; BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; TS018af012=0167a1c86195f9e6113f15cda0f27acd06889f7687265062cb86097f5a4b1cbed6be85a03ab5f8dce26099dc651d5eb4f099d35463e7c794db13e9b3c070e346c6b3d6df85; f5avraaaaaaaaaaaaaaaa_session_=PCIDOHHCIABMCNAPMAENNFMKALBDGHEPACHNFIOEFINJGHPPHBBNJGGKFFBFFFNAPPIDDJNDKHNGLDAJNPJAEOBIPIPJBKAAKGAOLODDMKKLNDLGFNMPBFGFOGNIKMHN; XSRF-TOKEN=eyJpdiI6Ik84bkltTlZldFdkbXBhRlhKWmNMUkE9PSIsInZhbHVlIjoiaXJiQzl4QzYxdDVNYjMzcXgzL1B1WHJjZVkrVDkwdGE1WmdPOG1EK3dsSFk4NlZmenF6WHdxOWRXTjJXRFlWMjduSmptaEk5RUhuem44bFlOYWhtTE12dXV2T1FrTFhUV3loNGFvN01VSlZYMUlDNDUrY2w1OStyc25QcFJMRVkiLCJtYWMiOiJkYTFlY2VkMzlmNjU1NzUwNDNmNGQyMzIwM2E5MzcyNjk1OTA0ZWQxMGM5ODFhMTI4NzIxNjlkZGQwN2FlMjE5IiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6IlArbUUxNnF0RjJnR1BQQ0NnVkhVVGc9PSIsInZhbHVlIjoiMDlnMkI3NTk0ak9LekZWNlllYjhuelFvdDNjWnJUUW8wSWJ4bWhRbFR5STBqcHhEdCsybTZpNFlGaFM4Um0wWG5vN3hMcWVBK21rVkZGUFBrWEVxWjhoQ1ZkNzk2NkhlOXBpL3N4a1YyeHQ4VkV3OE14aTBkU1g3SWs3cFJWREciLCJtYWMiOiI1Nzg5ZGFmYjRmYzExZmRhMWVlNDU2NzcxMDdjMjgxYjcwMjY3NDJlODdlMjAxNTI4YTAwZWRiMWI5Zjg1YzYzIiwidGFnIjoiIn0%3D; TS0151fc2b=0167a1c86118d45785fd31603fab161c6619d4bedf2466c608ef78ed187fcca638455497665181a619bd1ce01ea165dee053fb76bf; TS1a53eee7027=0815dd1fcdab20008a2279bea573cf45d96709f9410736905970d09471e5359d58e67dc53e23e80508a5506749113000a277bdcccdddff1aa85dea369ddc0839e951d8f0a4152604d03bcb32585e4df0093c0c39fa629154a06431f2aa11386d',
    }

    data = {
        '_token': 'jrfe2sJNmH0kg1jD1lf30kYF0nU7qwamN89NdQe3',
        'start': '0',
        'length': '10',
        'nama_usaha': '',
        'alamat_usaha': '',
        'provinsi': '120',
        'kabupaten': '',
        'kecamatan': '',
        'desa': '',
        'status_filter': 'semua',
        'rtotal': '1',
        'sumber_data': 'tambahan-daerah',
        'skala_usaha': '',
        'idsbr': idsbr,
        'history_profiling': '',
        'f_latlong': '',
        'f_gc': '',
    }

    try:
        response = requests.post(
            'https://matchapro.web.bps.go.id/direktori-usaha/data-gc-card',
            cookies=cookies,
            headers=headers,
            data=data,
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                data = result['data'][0]
                perusahaan_id = data.get('perusahaan_id')
                latitude = data.get('latitude')
                longitude = data.get('longitude')
                latlong_status = data.get('latlong_status', 'invalid')
                
                print(f"✓ Found business ID for idsbr {idsbr}: {perusahaan_id[:20]}...")
                print(f"  Geocoding Status: {latlong_status}")
                print(f"  Coordinates: ({latitude}, {longitude})")
                
                return {
                    'perusahaan_id': perusahaan_id,
                    'latitude': latitude,
                    'longitude': longitude,
                    'latlong_status': latlong_status
                }
        
        print(f"✗ No data found for idsbr {idsbr}")
        print(f"  Response Status Code: {response.status_code}")
        print(f"  Response: {response.text}")
        return None
        
    except Exception as e:
        print(f"✗ Error getting business ID for idsbr {idsbr}: {str(e)}")
        return None


def update_business_status(perusahaan_id, status, final_latitude, final_longitude):
    """
    Second request: Update business status by perusahaan_id
    Uses specific cookies and headers for this request (DIFFERENT from first request)
    
    Args:
        perusahaan_id: The encrypted business ID from first request
        status: The new status (hasilgc value)
        final_latitude: The final latitude (already validated/transformed)
        final_longitude: The final longitude (already validated/transformed)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"  Using coordinates: ({final_latitude}, {final_longitude})")
    cookies = {
        '_ga_K98R6MSKRH': 'GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0',
        '_ga': 'GA1.3.1178577724.1767583475',
        '_ga_XXTTVXWHDB': 'GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0',
        'BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool': '1418199050.20480.0000',
        'TS018af012': '0167a1c86195f9e6113f15cda0f27acd06889f7687265062cb86097f5a4b1cbed6be85a03ab5f8dce26099dc651d5eb4f099d35463e7c794db13e9b3c070e346c6b3d6df85',
        'f5avraaaaaaaaaaaaaaaa_session_': 'PCIDOHHCIABMCNAPMAENNFMKALBDGHEPACHNFIOEFINJGHPPHBBNJGGKFFBFFFNAPPIDDJNDKHNGLDAJNPJAEOBIPIPJBKAAKGAOLODDMKKLNDLGFNMPBFGFOGNIKMHN',
        'TS0151fc2b': '0167a1c86118d45785fd31603fab161c6619d4bedf2466c608ef78ed187fcca638455497665181a619bd1ce01ea165dee053fb76bf',
        'XSRF-TOKEN': 'eyJpdiI6InQyZTlrTFQzdXUwN01hektMS01SbHc9PSIsInZhbHVlIjoidHhnN0l1NUNmMnRQc2NobHlDNEp2MzIwQWY5MFl0VXdkamp2bGgzMU81SzhtdHFsZC9zOThJNFd2MFNSSU5BWjRIZ3MyK2FCQjlueDRiOEVEQ1ZRZkFBS3BlZVRuWnpBWW1leG00VC9aTlN6MmdWQWZQT25CTFVsSWNibVJzcVoiLCJtYWMiOiIzOTRlOGQzZjVlODEwMjFhZTQ5YzFmZmFjZTg3OTlmM2MzN2ZkNmQwNGIzZDAzMmFkMjFlZGIxMTZkODk5ZTFhIiwidGFnIjoiIn0%3D',
        'laravel_session': 'eyJpdiI6IjEvZ1lDVGRsajdRdHlOc0xqVUswd3c9PSIsInZhbHVlIjoiVm80ZFArdlhtT3lGdUxwakVrbHJUbWg5c3Mzb2hGVWFzZEZlcElUeUVwdnNwYlkvQkcxanhMZ3JVRENBSWFYMndCK3JzU2NxNlhYMmVONFo0WllGMFF0S201Q3VXUUJJbTBNcVRVS0RBbFlFam40eXZLY2lRUm54bzR3SEtEZnEiLCJtYWMiOiIwNGI4MTVjNWQ2MDc4MmNjOWE3OWQ4ZDJhOGE3NTg1NGMzOWQyZmIzMWIwMTViMmJlNzdjNDE3NWRkMjFlZGZkIiwidGFnIjoiIn0%3D',
        'TS1a53eee7027': '0815dd1fcdab200096b7af70ac8713969ad986c483395f453f47b2734156d7e9a0b3bc0c65b6ffa608e39d2a90113000f329b19befbd720da557f4ab0d59e7684e7afcc458af57801932696e38b99dc867ace0f9a9904dfb76e0c6bc2a413ebc',
    }

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'https://matchapro.web.bps.go.id',
        'Referer': 'https://matchapro.web.bps.go.id/dirgc',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        # 'Cookie': '_ga_K98R6MSKRH=GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0; _ga=GA1.3.1178577724.1767583475; _ga_XXTTVXWHDB=GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0; BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; TS018af012=0167a1c86195f9e6113f15cda0f27acd06889f7687265062cb86097f5a4b1cbed6be85a03ab5f8dce26099dc651d5eb4f099d35463e7c794db13e9b3c070e346c6b3d6df85; f5avraaaaaaaaaaaaaaaa_session_=PCIDOHHCIABMCNAPMAENNFMKALBDGHEPACHNFIOEFINJGHPPHBBNJGGKFFBFFFNAPPIDDJNDKHNGLDAJNPJAEOBIPIPJBKAAKGAOLODDMKKLNDLGFNMPBFGFOGNIKMHN; TS0151fc2b=0167a1c86118d45785fd31603fab161c6619d4bedf2466c608ef78ed187fcca638455497665181a619bd1ce01ea165dee053fb76bf; XSRF-TOKEN=eyJpdiI6InQyZTlrTFQzdXUwN01hektMS01SbHc9PSIsInZhbHVlIjoidHhnN0l1NUNmMnRQc2NobHlDNEp2MzIwQWY5MFl0VXdkamp2bGgzMU81SzhtdHFsZC9zOThJNFd2MFNSSU5BWjRIZ3MyK2FCQjlueDRiOEVEQ1ZRZkFBS3BlZVRuWnpBWW1leG00VC9aTlN6MmdWQWZQT25CTFVsSWNibVJzcVoiLCJtYWMiOiIzOTRlOGQzZjVlODEwMjFhZTQ5YzFmZmFjZTg3OTlmM2MzN2ZkNmQwNGIzZDAzMmFkMjFlZGIxMTZkODk5ZTFhIiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6IjEvZ1lDVGRsajdRdHlOc0xqVUswd3c9PSIsInZhbHVlIjoiVm80ZFArdlhtT3lGdUxwakVrbHJUbWg5c3Mzb2hGVWFzZEZlcElUeUVwdnNwYlkvQkcxanhMZ3JVRENBSWFYMndCK3JzU2NxNlhYMmVONFo0WllGMFF0S201Q3VXUUJJbTBNcVRVS0RBbFlFam40eXZLY2lRUm54bzR3SEtEZnEiLCJtYWMiOiIwNGI4MTVjNWQ2MDc4MmNjOWE3OWQ4ZDJhOGE3NTg1NGMzOWQyZmIzMWIwMTViMmJlNzdjNDE3NWRkMjFlZGZkIiwidGFnIjoiIn0%3D; TS1a53eee7027=0815dd1fcdab200096b7af70ac8713969ad986c483395f453f47b2734156d7e9a0b3bc0c65b6ffa608e39d2a90113000f329b19befbd720da557f4ab0d59e7684e7afcc458af57801932696e38b99dc867ace0f9a9904dfb76e0c6bc2a413ebc',
    }
 
    try:
        data = {
            'perusahaan_id': perusahaan_id,
            'latitude': final_latitude,
            'longitude': final_longitude,
            'hasilgc': str(status),
            '_token': 'jrfe2sJNmH0kg1jD1lf30kYF0nU7qwamN89NdQe3',
        }

        response = requests.post('https://matchapro.web.bps.go.id/dirgc/konfirmasi-user', cookies=cookies, headers=headers, data=data) 
        
        print(f"Update Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ Successfully updated status to {status}")
            return True
        else:
            print(f"✗ Failed to update status. Status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error updating business status: {str(e)}")
        return False

def process_gc_excel():
    """
    Main function: Read Excel file and process rows with empty or failed response
    """
    try:
        # Read Excel file with specific dtypes to ensure columns are read as strings
        df = pd.read_excel(EXCEL_FILE, dtype={
            'idsbr': str, 
            'status': str, 
            'response': str,
            'latlong_status_initial': str,
            'latitude_initial': str,
            'longitude_initial': str,
            'latitude_final': str,
            'longitude_final': str
        })
        print(f"\n✓ Excel file loaded. Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Filter rows where response is empty or failed (not yet processed or retry failed)
        empty_response_rows = df[(df['response'].isna() | (df['response'] == '') | (df['response'] == 'failed'))]
        print(f"\nRows with empty or failed response: {len(empty_response_rows)}")
        
        if len(empty_response_rows) == 0:
            print("No rows with empty or failed response found.")
            return
        
        # Process each row
        for idx, row in empty_response_rows.iterrows():
            print(f"\n--- Processing row {idx + 1} ---")
            idsbr = str(row['idsbr']).strip()
            status = str(row['status']).strip()
            
            print(f"IDSBR: {idsbr}, Status (hasilgc): {status}")
            
            # Step 1: Get business ID by idsbr (also gets geocoding data)
            result = get_business_id_by_idsbr(idsbr)
            
            if not result:
                df.at[idx, 'response'] = 'failed'
                print("Response: failed (could not get business ID)")
                continue
            
            # Store initial coordinate information
            df.at[idx, 'latlong_status_initial'] = str(result['latlong_status']) if result['latlong_status'] is not None else None
            df.at[idx, 'latitude_initial'] = str(result['latitude']) if result['latitude'] is not None else None
            df.at[idx, 'longitude_initial'] = str(result['longitude']) if result['longitude'] is not None else None
            
            # Step 2: Transform coordinates if needed (only call once)
            if result['latlong_status'] == 'valid':
                final_latitude = result['latitude']
                final_longitude = result['longitude']
                print(f"  Using valid geocoded coordinates")
            else:
                print(f"  Geocoding status '{result['latlong_status']}' is not valid, transforming coordinates...")
                try:
                    final_latitude, final_longitude = transform_coordinates(result['latitude'], result['longitude'])
                    print(f"  Transformed coordinates: ({final_latitude}, {final_longitude})")
                except Exception as e:
                    print(f"  ✗ Error transforming coordinates: {str(e)}")
                    df.at[idx, 'response'] = 'coordinate_invalid'
                    print("Response: coordinate_invalid (transformation failed)")
                    continue
            
            # Store final coordinate information
            df.at[idx, 'latitude_final'] = str(final_latitude) if final_latitude is not None else None
            df.at[idx, 'longitude_final'] = str(final_longitude) if final_longitude is not None else None
            
            # Add delay between first and second request
            time.sleep(1)
            
            # Step 3: Update business status with the transformed coordinates
            update_result = update_business_status(
                result['perusahaan_id'],
                status,
                final_latitude,
                final_longitude
            )
            if update_result:
                df.at[idx, 'response'] = 'success'
            else:
                df.at[idx, 'response'] = 'failed'
            
            # Save Excel after each row update
            df.to_excel(EXCEL_FILE, index=False)
            print(f"✓ Row {idx + 1} saved to Excel")
        
        print(f"\n✓ All rows processed and saved to {EXCEL_FILE}")
        
    except FileNotFoundError:
        print(f"✗ Excel file not found at {EXCEL_FILE}")
    except Exception as e:
        print(f"✗ Error processing Excel file: {str(e)}")


if __name__ == "__main__":
    process_gc_excel()
