import requests
import pandas as pd
import time
from pathlib import Path

# Configuration
EXCEL_FILE = Path('source_gc_update/gc.xlsx')
BASE_URL = 'https://matchapro.web.bps.go.id'
INITIAL_GC_TOKEN = 'JRIaYqdBn0OKXkSy0vWFfbez1VGvXycWBILg2gy3'  # Initial GC token for first request


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
        Dictionary with perusahaan_id, latitude, longitude, latlong_status, gc_username if found, None otherwise
    """
    cookies = {
        '_ga': 'GA1.3.451362273.1753614616',
        'cf_clearance': 'dGpYKScmzan70LrF36.U20WHMIKq1Wj576cTZOmd2.M-1754396822-1.2.1.1-EPp6pVOKOLonPCwEi0aRWlGcgf53JXZeG5wD1nob3Ca_Ho1dfrd.IO69mzanDsBOIY8x_EAdAECfs_PPK7OwnXCNDGcdfNHVpSPBihcOqa4mZOKBb7vIBZBQjTvAPP93nf00eqySWGtkoYFWpBLl4c1MhB8Z5XskX3G4NShRjyJQv6b7zxnWvoO1MGWvYS.73u6DZ6IgwYnoWh8KKrVbzYaZWru4KhB4Zfx7Pn_b2NY',
        '_ga_XXTTVXWHDB': 'GS2.3.s1754410359$o3$g0$t1754410359$j60$l0$h0',
        'BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool': '1418199050.20480.0000',
        'f5avraaaaaaaaaaaaaaaa_session_': 'NCBOEEOIOPHDLPHNHGMOABNNBNILCKCPMOOIOCIMHPJBJOCIAALIKFBOCNGKJHBENNLDJMNHMPCPHALIEEEAKBEMNOBENFEPPBJOECKAPFDGMOOAKOBFDDEDJNHJPMAM',
        'TS0151fc2b': '0167a1c8617fc53c7c1282bff2f6a5dbe1ff1decc2098b903426a66a10f133e61654f911977934f5e35941b1980176040d4aa2fffa',
        'XSRF-TOKEN': 'eyJpdiI6Ii9qUWpNSkJQY09XSGYyVUJiNkJWYlE9PSIsInZhbHVlIjoiNkFGWlZ6SGdWYnc2Zm5aeTVRaUFmODltQ1B2eEdMSlJnRys3bTc5Q2hPaGRVaFVWVmYvNzB6bC82cmhYMHpBbnFqTUNxQXRBT0txeWlVV1FlMmZubkxoeFowY1ovTGpFZHo3cGRJK2RjTm9qWnJGeEZIWHVPQkhobEljSzdBQXYiLCJtYWMiOiJkN2YzZGUxYTE3ZDdhYTEwNDM3MDNjZjMwYmEyMWZiMTRkMzJjODY5ZTZkMmMzOWFkZjBjN2UzZDkxNTY0MmMzIiwidGFnIjoiIn0%3D',
        'laravel_session': 'eyJpdiI6IjF5YWFVNkw5WkFHVGVuaVRNZHFqcUE9PSIsInZhbHVlIjoieGREOFkybzlIZnRFRVV6bUpuTjI3M3JLT080RWlIVm9BSUdJV3RDeWhxb0tVdUFyZUYwSlByWStCK0FaWllnZjBSTmp6NUxiZWdxdTB6OGZwOVM1a3JEQ1dkS0RUWU9BdE5lR1dSWElvenRQVTdOQXBvd0VYYk12SUxLcUdrU1YiLCJtYWMiOiIxZjk4OWM2ZDZhZjMxOWUwM2U4Yjk5OGVjNzQ4ZDUwMGY3NzQxZmVhY2JiZWMyOGEyNmZhYzU0MzQ1ZDJiYWFmIiwidGFnIjoiIn0%3D',
        'TS1a53eee7027': '0815dd1fcdab20001f28c692e7b65c8a46600d239d6b702860fdc30f60e97ac0c993eef29b9357de0827815cb41130001a5ec517101f1e1f8878e1808d97cb60f1aa76090aaf9703ca5d030baa67ea1f9a36200616ea90fb9edd6980a6bfb786',
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
        # 'Cookie': '_ga=GA1.3.451362273.1753614616; cf_clearance=dGpYKScmzan70LrF36.U20WHMIKq1Wj576cTZOmd2.M-1754396822-1.2.1.1-EPp6pVOKOLonPCwEi0aRWlGcgf53JXZeG5wD1nob3Ca_Ho1dfrd.IO69mzanDsBOIY8x_EAdAECfs_PPK7OwnXCNDGcdfNHVpSPBihcOqa4mZOKBb7vIBZBQjTvAPP93nf00eqySWGtkoYFWpBLl4c1MhB8Z5XskX3G4NShRjyJQv6b7zxnWvoO1MGWvYS.73u6DZ6IgwYnoWh8KKrVbzYaZWru4KhB4Zfx7Pn_b2NY; _ga_XXTTVXWHDB=GS2.3.s1754410359$o3$g0$t1754410359$j60$l0$h0; BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; f5avraaaaaaaaaaaaaaaa_session_=NCBOEEOIOPHDLPHNHGMOABNNBNILCKCPMOOIOCIMHPJBJOCIAALIKFBOCNGKJHBENNLDJMNHMPCPHALIEEEAKBEMNOBENFEPPBJOECKAPFDGMOOAKOBFDDEDJNHJPMAM; TS0151fc2b=0167a1c8617fc53c7c1282bff2f6a5dbe1ff1decc2098b903426a66a10f133e61654f911977934f5e35941b1980176040d4aa2fffa; XSRF-TOKEN=eyJpdiI6Ii9qUWpNSkJQY09XSGYyVUJiNkJWYlE9PSIsInZhbHVlIjoiNkFGWlZ6SGdWYnc2Zm5aeTVRaUFmODltQ1B2eEdMSlJnRys3bTc5Q2hPaGRVaFVWVmYvNzB6bC82cmhYMHpBbnFqTUNxQXRBT0txeWlVV1FlMmZubkxoeFowY1ovTGpFZHo3cGRJK2RjTm9qWnJGeEZIWHVPQkhobEljSzdBQXYiLCJtYWMiOiJkN2YzZGUxYTE3ZDdhYTEwNDM3MDNjZjMwYmEyMWZiMTRkMzJjODY5ZTZkMmMzOWFkZjBjN2UzZDkxNTY0MmMzIiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6IjF5YWFVNkw5WkFHVGVuaVRNZHFqcUE9PSIsInZhbHVlIjoieGREOFkybzlIZnRFRVV6bUpuTjI3M3JLT080RWlIVm9BSUdJV3RDeWhxb0tVdUFyZUYwSlByWStCK0FaWllnZjBSTmp6NUxiZWdxdTB6OGZwOVM1a3JEQ1dkS0RUWU9BdE5lR1dSWElvenRQVTdOQXBvd0VYYk12SUxLcUdrU1YiLCJtYWMiOiIxZjk4OWM2ZDZhZjMxOWUwM2U4Yjk5OGVjNzQ4ZDUwMGY3NzQxZmVhY2JiZWMyOGEyNmZhYzU0MzQ1ZDJiYWFmIiwidGFnIjoiIn0%3D; TS1a53eee7027=0815dd1fcdab20001f28c692e7b65c8a46600d239d6b702860fdc30f60e97ac0c993eef29b9357de0827815cb41130001a5ec517101f1e1f8878e1808d97cb60f1aa76090aaf9703ca5d030baa67ea1f9a36200616ea90fb9edd6980a6bfb786',
    }

    data = {
        '_token': 'Frnxf8hxS0xbacAaw0RLFtjA0JirphKehu0mUp5d',
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
                gc_username = data.get('gc_username')
                
                print(f"✓ Found business ID for idsbr {idsbr}: {perusahaan_id[:20]}...")
                print(f"  Geocoding Status: {latlong_status}")
                print(f"  Coordinates: ({latitude}, {longitude})")
                print(f"  GC Username: {gc_username if gc_username else 'Not yet geocoded'}")
                
                return {
                    'perusahaan_id': perusahaan_id,
                    'latitude': latitude,
                    'longitude': longitude,
                    'latlong_status': latlong_status,
                    'gc_username': gc_username
                }
        
        print(f"✗ No data found for idsbr {idsbr}")
        print(f"  Response Status Code: {response.status_code}")
        print(f"  Response: {response.text}")
        return None
        
    except Exception as e:
        print(f"✗ Error getting business ID for idsbr {idsbr}: {str(e)}")
        return None


def update_business_status(perusahaan_id, status, final_latitude, final_longitude, gc_token):
    """
    Second request: Update business status by perusahaan_id
    Uses specific cookies and headers for this request (DIFFERENT from first request)
    
    Args:
        perusahaan_id: The encrypted business ID from first request
        status: The new status (hasilgc value)
        final_latitude: The final latitude (already validated/transformed)
        final_longitude: The final longitude (already validated/transformed)
        gc_token: The GC token to use for this request
        
    Returns:
        Tuple of (success: bool, new_gc_token: str or None)
    """
    print(f"  Using coordinates: ({final_latitude}, {final_longitude})")
    cookies = {
        '_ga': 'GA1.3.451362273.1753614616',
        'cf_clearance': 'dGpYKScmzan70LrF36.U20WHMIKq1Wj576cTZOmd2.M-1754396822-1.2.1.1-EPp6pVOKOLonPCwEi0aRWlGcgf53JXZeG5wD1nob3Ca_Ho1dfrd.IO69mzanDsBOIY8x_EAdAECfs_PPK7OwnXCNDGcdfNHVpSPBihcOqa4mZOKBb7vIBZBQjTvAPP93nf00eqySWGtkoYFWpBLl4c1MhB8Z5XskX3G4NShRjyJQv6b7zxnWvoO1MGWvYS.73u6DZ6IgwYnoWh8KKrVbzYaZWru4KhB4Zfx7Pn_b2NY',
        '_ga_XXTTVXWHDB': 'GS2.3.s1754410359$o3$g0$t1754410359$j60$l0$h0',
        'BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool': '1418199050.20480.0000',
        'f5avraaaaaaaaaaaaaaaa_session_': 'NCBOEEOIOPHDLPHNHGMOABNNBNILCKCPMOOIOCIMHPJBJOCIAALIKFBOCNGKJHBENNLDJMNHMPCPHALIEEEAKBEMNOBENFEPPBJOECKAPFDGMOOAKOBFDDEDJNHJPMAM',
        'TS0151fc2b': '0167a1c8617fc53c7c1282bff2f6a5dbe1ff1decc2098b903426a66a10f133e61654f911977934f5e35941b1980176040d4aa2fffa',
        'XSRF-TOKEN': 'eyJpdiI6IlF1bE9vYnRrbHkxZU5TOWZ5Z0xWRVE9PSIsInZhbHVlIjoiYUJDQW1NZWJoaEdrSUxVYloweEpWcG9JSEZFRHkrMUQ1TDhaeCtnbDQvaUExY0Z0KzQrbytSeGhCZDQ2NklCVWlZRURpMEQxdUEvTVJYMUJuUkgrVXJrWnFJbHJhaEVtNnYxUXlLcmFuNUVVR25WcVJQWG0yV3ZQVXJZNy9jYVoiLCJtYWMiOiI2ZjllNWFlMTA0ZTUzMDFjMjViNzJlZmU1NWE5MDYwOTRkMjQxZGY5YjJjYTUyMzE4MjBkODk4ODRlZGUzYzM2IiwidGFnIjoiIn0%3D',
        'laravel_session': 'eyJpdiI6ImN6Mlg5eDNVTW1KZk1PSU0wTGtsZUE9PSIsInZhbHVlIjoiL09FOThjRHAxTUpJUFk4MU83Ky9Xd1h4dk9QemZ4UkhYSHN0WFVXTEw3Z1lBWmFvK0JsY0NYdzlqZUNpaURuSytJQjBNaDIzaExjRmgwdVRZL1drS2YrNmdLUHd4Q1QwRnppaG5wWSt5L2RGRmJjOGluNnFTbkdUNkN1YS9MNU8iLCJtYWMiOiI1NmUwMmI4MjRhNzQ5OTY4MjM0NzQ3ODVkNjY0YTQyOWQzMjZjNGMyMmE2MGJmZmY2M2I3YzkwYzQ2YzY2MjFjIiwidGFnIjoiIn0%3D',
        'TS1a53eee7027': '0815dd1fcdab200009af98de45912f61835036c7c45d0fbe18a62c87441e48c48209bc003912fbc308d5b293ee11300085fbb2f6ac31713edbfd04395fd4a386d214ab831a663cba2dfd895dcf96c8e8c8773a5a9e3df584a0e646969cdbf2a7',
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
        # 'Cookie': '_ga=GA1.3.451362273.1753614616; cf_clearance=dGpYKScmzan70LrF36.U20WHMIKq1Wj576cTZOmd2.M-1754396822-1.2.1.1-EPp6pVOKOLonPCwEi0aRWlGcgf53JXZeG5wD1nob3Ca_Ho1dfrd.IO69mzanDsBOIY8x_EAdAECfs_PPK7OwnXCNDGcdfNHVpSPBihcOqa4mZOKBb7vIBZBQjTvAPP93nf00eqySWGtkoYFWpBLl4c1MhB8Z5XskX3G4NShRjyJQv6b7zxnWvoO1MGWvYS.73u6DZ6IgwYnoWh8KKrVbzYaZWru4KhB4Zfx7Pn_b2NY; _ga_XXTTVXWHDB=GS2.3.s1754410359$o3$g0$t1754410359$j60$l0$h0; BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; f5avraaaaaaaaaaaaaaaa_session_=NCBOEEOIOPHDLPHNHGMOABNNBNILCKCPMOOIOCIMHPJBJOCIAALIKFBOCNGKJHBENNLDJMNHMPCPHALIEEEAKBEMNOBENFEPPBJOECKAPFDGMOOAKOBFDDEDJNHJPMAM; TS0151fc2b=0167a1c8617fc53c7c1282bff2f6a5dbe1ff1decc2098b903426a66a10f133e61654f911977934f5e35941b1980176040d4aa2fffa; XSRF-TOKEN=eyJpdiI6IlF1bE9vYnRrbHkxZU5TOWZ5Z0xWRVE9PSIsInZhbHVlIjoiYUJDQW1NZWJoaEdrSUxVYloweEpWcG9JSEZFRHkrMUQ1TDhaeCtnbDQvaUExY0Z0KzQrbytSeGhCZDQ2NklCVWlZRURpMEQxdUEvTVJYMUJuUkgrVXJrWnFJbHJhaEVtNnYxUXlLcmFuNUVVR25WcVJQWG0yV3ZQVXJZNy9jYVoiLCJtYWMiOiI2ZjllNWFlMTA0ZTUzMDFjMjViNzJlZmU1NWE5MDYwOTRkMjQxZGY5YjJjYTUyMzE4MjBkODk4ODRlZGUzYzM2IiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6ImN6Mlg5eDNVTW1KZk1PSU0wTGtsZUE9PSIsInZhbHVlIjoiL09FOThjRHAxTUpJUFk4MU83Ky9Xd1h4dk9QemZ4UkhYSHN0WFVXTEw3Z1lBWmFvK0JsY0NYdzlqZUNpaURuSytJQjBNaDIzaExjRmgwdVRZL1drS2YrNmdLUHd4Q1QwRnppaG5wWSt5L2RGRmJjOGluNnFTbkdUNkN1YS9MNU8iLCJtYWMiOiI1NmUwMmI4MjRhNzQ5OTY4MjM0NzQ3ODVkNjY0YTQyOWQzMjZjNGMyMmE2MGJmZmY2M2I3YzkwYzQ2YzY2MjFjIiwidGFnIjoiIn0%3D; TS1a53eee7027=0815dd1fcdab200009af98de45912f61835036c7c45d0fbe18a62c87441e48c48209bc003912fbc308d5b293ee11300085fbb2f6ac31713edbfd04395fd4a386d214ab831a663cba2dfd895dcf96c8e8c8773a5a9e3df584a0e646969cdbf2a7',
    }
 
    try:
        data = {
            'perusahaan_id': perusahaan_id,
            'latitude': final_latitude,
            'longitude': final_longitude,
            'hasilgc': str(status),
            '_token': 'Frnxf8hxS0xbacAaw0RLFtjA0JirphKehu0mUp5d',
            'gc_token': gc_token,
        }

        response = requests.post('https://matchapro.web.bps.go.id/dirgc/konfirmasi-user', cookies=cookies, headers=headers, data=data) 
        
        print(f"Update Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            new_gc_token = result.get('new_gc_token')
            
            if result.get('status') == 'success':
                print(f"✓ Successfully updated status to {status}")
                print(f"  New GC Token: {new_gc_token[:20]}..." if new_gc_token else "  No new GC token received")
                return True, new_gc_token
            else:
                print(f"✗ Update failed: {result.get('message', 'Unknown error')}")
                return False, gc_token
        else:
            print(f"✗ Failed to update status. Status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False, gc_token
            
    except Exception as e:
        print(f"✗ Error updating business status: {str(e)}")
        return False, gc_token

def process_gc_excel():
    """
    Main function: Read Excel file and process rows with empty or failed response
    """
    # Initialize GC token with the initial value
    current_gc_token = INITIAL_GC_TOKEN
    
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
            
            # Check if already geocoded by another user
            if result.get('gc_username'):
                df.at[idx, 'response'] = 'already_updated'
                print(f"Response: already_updated (geocoded by user: {result.get('gc_username')})")
                # Save and skip to next row
                df.to_excel(EXCEL_FILE, index=False)
                print(f"✓ Row {idx + 1} saved to Excel")
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
            update_success, new_gc_token = update_business_status(
                result['perusahaan_id'],
                status,
                final_latitude,
                final_longitude,
                current_gc_token
            )
            
            if update_success:
                df.at[idx, 'response'] = 'success'
                # Update current_gc_token for next request if new token received
                if new_gc_token:
                    current_gc_token = new_gc_token
                    print(f"  ✓ GC token updated for next request")
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
