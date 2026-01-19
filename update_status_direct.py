import requests
import pandas as pd
import time
from pathlib import Path

# Configuration
EXCEL_FILE = Path('source_gc_update/gc.xlsx')
BASE_URL = 'https://matchapro.web.bps.go.id'
INITIAL_GC_TOKEN = 'pWozsCXmpPJ6Maq53m5AJo31NvFGczaEeJt4AWwi'  # Initial GC token for first request


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
        '_ga_K98R6MSKRH': 'GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0',
        '_ga': 'GA1.3.1178577724.1767583475',
        '_ga_XXTTVXWHDB': 'GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0',
        'BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool': '1418199050.20480.0000',
        'f5avraaaaaaaaaaaaaaaa_session_': 'KPJMAIFBFOLJBEJAJDKPPAJJHJNKOFPJLDOLDDMJBHCCLLJDGHBKJLDAFBLDGKPEIKLDAFEFGLOPKNMPPGIAALPIBHPIIMPMKEGPILKMPAEAEPJDOOKJNCBGLAHHJBAO',
        'TS0151fc2b': '0167a1c8619519b04d836492ee0b3bc505ad4a21064d6c63688ca95ce1fbe09eff5b0fbbd3084827cf4aac5785f6fcf9bf17fbffc5',
        'XSRF-TOKEN': 'eyJpdiI6IlY0eW91dXQ4NDNqNjYrVFFJakM4amc9PSIsInZhbHVlIjoiR29ocWtWRHNwYmtuU3BmWDAvVUhrNU9hLzNzWSswZjlpY3dHVDdlaG16RUpsTWhKSUcyd1Ewc1dTWldNS2p0VnUxWWlvVDh6dWt1OE50Sis0dWx3NU81YkRMRXU4WkR4ZjY5aHNXMnlBRmVoRS9JSk5hWW9xMmFaUitrT01HR2oiLCJtYWMiOiIxNGY4MDMxMGQzYjMxZTA5YWIwYWFiYWJiNzFhZGIzNTMyMWFiMTk1MDQyNTlmNDk2MjdkOWRhMmI2OGFmMjljIiwidGFnIjoiIn0%3D',
        'laravel_session': 'eyJpdiI6IkhuTWloQmY1UnNyMlpNWDBaRTJ0ekE9PSIsInZhbHVlIjoiVGNUOUpMOSsybGM2UVZJSFZ6NUlzYWs4RjNicDZoWmNURGw2Mm9Cak9tY2Y4Um1NcEtoYXZ1VmlvTElYU2FBZEpsOHdOcDZscjg1NzFnMmFaODFsbzF1bllpTWtqWkNVS1p5Vk1iSTJEM1pxRkxyVTVtU3ZCbFJXWFdKRUlaMnEiLCJtYWMiOiJmNWQ2OTBiOTQ4NGVhZTRkNDg0N2VmZDkyZjlmM2JiYjQ0ZTRmYmRkN2QzNmFjMzYwZjBhOTM3YTU5Yzg3MWY2IiwidGFnIjoiIn0%3D',
        'TS1a53eee7027': '0815dd1fcdab2000a3fa50d7638a0587480f25d2ae83981a5336393517ddb14c54dd7857690453b308228b2b011130002590508dea5385c0aee79e71979a316081c3c9dabb10627b9c1b2def287984da447cdf4c31080a1026b3ba67379d189e',
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
        'User-Agent': 'Mozilla/5.0 (Linux; Android 12; M2010J19CG Build/SKQ1.211202.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/143.0.7499.192 Mobile Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': 'Android WebView',
        'sec-ch-ua-mobile': '1',
        'sec-ch-ua-platform': 'Android',
        # 'Cookie': '_ga_K98R6MSKRH=GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0; _ga=GA1.3.1178577724.1767583475; _ga_XXTTVXWHDB=GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0; BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; f5avraaaaaaaaaaaaaaaa_session_=KPJMAIFBFOLJBEJAJDKPPAJJHJNKOFPJLDOLDDMJBHCCLLJDGHBKJLDAFBLDGKPEIKLDAFEFGLOPKNMPPGIAALPIBHPIIMPMKEGPILKMPAEAEPJDOOKJNCBGLAHHJBAO; TS0151fc2b=0167a1c8619519b04d836492ee0b3bc505ad4a21064d6c63688ca95ce1fbe09eff5b0fbbd3084827cf4aac5785f6fcf9bf17fbffc5; XSRF-TOKEN=eyJpdiI6IlY0eW91dXQ4NDNqNjYrVFFJakM4amc9PSIsInZhbHVlIjoiR29ocWtWRHNwYmtuU3BmWDAvVUhrNU9hLzNzWSswZjlpY3dHVDdlaG16RUpsTWhKSUcyd1Ewc1dTWldNS2p0VnUxWWlvVDh6dWt1OE50Sis0dWx3NU81YkRMRXU4WkR4ZjY5aHNXMnlBRmVoRS9JSk5hWW9xMmFaUitrT01HR2oiLCJtYWMiOiIxNGY4MDMxMGQzYjMxZTA5YWIwYWFiYWJiNzFhZGIzNTMyMWFiMTk1MDQyNTlmNDk2MjdkOWRhMmI2OGFmMjljIiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6IkhuTWloQmY1UnNyMlpNWDBaRTJ0ekE9PSIsInZhbHVlIjoiVGNUOUpMOSsybGM2UVZJSFZ6NUlzYWs4RjNicDZoWmNURGw2Mm9Cak9tY2Y4Um1NcEtoYXZ1VmlvTElYU2FBZEpsOHdOcDZscjg1NzFnMmFaODFsbzF1bllpTWtqWkNVS1p5Vk1iSTJEM1pxRkxyVTVtU3ZCbFJXWFdKRUlaMnEiLCJtYWMiOiJmNWQ2OTBiOTQ4NGVhZTRkNDg0N2VmZDkyZjlmM2JiYjQ0ZTRmYmRkN2QzNmFjMzYwZjBhOTM3YTU5Yzg3MWY2IiwidGFnIjoiIn0%3D; TS1a53eee7027=0815dd1fcdab2000a3fa50d7638a0587480f25d2ae83981a5336393517ddb14c54dd7857690453b308228b2b011130002590508dea5385c0aee79e71979a316081c3c9dabb10627b9c1b2def287984da447cdf4c31080a1026b3ba67379d189e',
    }

    data = {
        '_token': 'rCPUh2tLIxBojPSDAv3V916v4SlpmoB45uix2WoB',
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
        '_ga_K98R6MSKRH': 'GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0',
        '_ga': 'GA1.3.1178577724.1767583475',
        '_ga_XXTTVXWHDB': 'GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0',
        'BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool': '1418199050.20480.0000',
        'f5avraaaaaaaaaaaaaaaa_session_': 'KPJMAIFBFOLJBEJAJDKPPAJJHJNKOFPJLDOLDDMJBHCCLLJDGHBKJLDAFBLDGKPEIKLDAFEFGLOPKNMPPGIAALPIBHPIIMPMKEGPILKMPAEAEPJDOOKJNCBGLAHHJBAO',
        'TS018af012': '0167a1c861b08e9862c2f162e7f6c59c59b2cd704eb0dda5f3ab514a28ecc2971f8fb9223ef219621b9ceceed099798516e3f1a60baa40bfff9125cb57320cd2bf9796c0df83a1780802612271340631dbd6c5b14d',
        'XSRF-TOKEN': 'eyJpdiI6Iis1U0xidmE3VkhWdDVKeHJBTTEzK0E9PSIsInZhbHVlIjoiQ1ZzemxraUxmZEthMFQwWktMMkZlbkRjQTNKbHJlVzYya0lubmRHeEt1VnJtaC9DanMwVitjLytFTjZhWG1weTQ5Zm9ZT0ViQ1pNMS8vcm5peGIxMjJRMTZJK1dKcXhTWWdJbDk1aWcxSkpwelpucHNHdlRUSjlDU3NCT0FMQVUiLCJtYWMiOiIxN2I0ZTEzNWQ4ZWZiOGMwYzc3YTM1OTM4ZjdmZjgzYTdkODMzOGViNjMwOGE5MWY4NzMwN2MwMjEwNjVhYjc3IiwidGFnIjoiIn0%3D',
        'laravel_session': 'eyJpdiI6IlliUHluS3k0VmJCNDFpaHRvZTA4WHc9PSIsInZhbHVlIjoiR3pWazNLN1JEbFkwbmkwUkdTN2paWk5JTWZWOCtSV1JIVEdSQnNHTjdTeldMSzFKYXV5cngwN09ieEQrMEEzellHczJTcGcyWHUvUEtWTGhrQ0dpeUEvZ1gyQ1ExSzJaZThZR0FKRFdCeUd2cUNzdW93VnVNTndKNUU1L2w3Q0QiLCJtYWMiOiIwNzk3MjY2Mjk2OGQ3MDNlZmE5OGJiMWM1NGU1Y2Y4MjZiMTFjZTEwNGUzOGI3NjlhNGYzNjBlOGEyN2ZkNGUyIiwidGFnIjoiIn0%3D',
        'TS0151fc2b': '0167a1c8611f94b9fba0d6c75f5a4640b0fde676b860b6d4e4404aab9eacfbed4d7bea1a6e2abd6e39aadcf8cdcc58b7d45a80254c',
        'TS1a53eee7027': '0815dd1fcdab20000a2a533c837b403714b761b44e479e7e5640e555949c0fef37e6f7da76743e8d08e740095b113000479cdc606e0e7c439911fc96c9cfa8c3ba898b081665bdc0a3f74433911a154e5f6ffa3eb31003613342aad2d19a88c7',
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
        'User-Agent': 'Mozilla/5.0 (Linux; Android 12; M2010J19CG Build/SKQ1.211202.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/143.0.7499.192 Mobile Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': 'Android WebView',
        'sec-ch-ua-mobile': '1',
        'sec-ch-ua-platform': 'Android',
        # 'Cookie': '_ga_K98R6MSKRH=GS2.1.s1768283730$o3$g0$t1768283730$j60$l0$h0; _ga=GA1.3.1178577724.1767583475; _ga_XXTTVXWHDB=GS2.3.s1768286968$o2$g0$t1768286983$j45$l0$h0; BIGipServeriapps_webhost_mars_443.app~iapps_webhost_mars_443_pool=1418199050.20480.0000; f5avraaaaaaaaaaaaaaaa_session_=KPJMAIFBFOLJBEJAJDKPPAJJHJNKOFPJLDOLDDMJBHCCLLJDGHBKJLDAFBLDGKPEIKLDAFEFGLOPKNMPPGIAALPIBHPIIMPMKEGPILKMPAEAEPJDOOKJNCBGLAHHJBAO; TS018af012=0167a1c861b08e9862c2f162e7f6c59c59b2cd704eb0dda5f3ab514a28ecc2971f8fb9223ef219621b9ceceed099798516e3f1a60baa40bfff9125cb57320cd2bf9796c0df83a1780802612271340631dbd6c5b14d; XSRF-TOKEN=eyJpdiI6Iis1U0xidmE3VkhWdDVKeHJBTTEzK0E9PSIsInZhbHVlIjoiQ1ZzemxraUxmZEthMFQwWktMMkZlbkRjQTNKbHJlVzYya0lubmRHeEt1VnJtaC9DanMwVitjLytFTjZhWG1weTQ5Zm9ZT0ViQ1pNMS8vcm5peGIxMjJRMTZJK1dKcXhTWWdJbDk1aWcxSkpwelpucHNHdlRUSjlDU3NCT0FMQVUiLCJtYWMiOiIxN2I0ZTEzNWQ4ZWZiOGMwYzc3YTM1OTM4ZjdmZjgzYTdkODMzOGViNjMwOGE5MWY4NzMwN2MwMjEwNjVhYjc3IiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6IlliUHluS3k0VmJCNDFpaHRvZTA4WHc9PSIsInZhbHVlIjoiR3pWazNLN1JEbFkwbmkwUkdTN2paWk5JTWZWOCtSV1JIVEdSQnNHTjdTeldMSzFKYXV5cngwN09ieEQrMEEzellHczJTcGcyWHUvUEtWTGhrQ0dpeUEvZ1gyQ1ExSzJaZThZR0FKRFdCeUd2cUNzdW93VnVNTndKNUU1L2w3Q0QiLCJtYWMiOiIwNzk3MjY2Mjk2OGQ3MDNlZmE5OGJiMWM1NGU1Y2Y4MjZiMTFjZTEwNGUzOGI3NjlhNGYzNjBlOGEyN2ZkNGUyIiwidGFnIjoiIn0%3D; TS0151fc2b=0167a1c8611f94b9fba0d6c75f5a4640b0fde676b860b6d4e4404aab9eacfbed4d7bea1a6e2abd6e39aadcf8cdcc58b7d45a80254c; TS1a53eee7027=0815dd1fcdab20000a2a533c837b403714b761b44e479e7e5640e555949c0fef37e6f7da76743e8d08e740095b113000479cdc606e0e7c439911fc96c9cfa8c3ba898b081665bdc0a3f74433911a154e5f6ffa3eb31003613342aad2d19a88c7',
    }
 
    try:
        data = {
            'perusahaan_id': perusahaan_id,
            'latitude': final_latitude,
            'longitude': final_longitude,
            'hasilgc': str(status),
            '_token': 'rCPUh2tLIxBojPSDAv3V916v4SlpmoB45uix2WoB',
            'edit_nama': '0',
            'edit_alamat': '0',
            'nama_usaha': '',
            'alamat_usaha': '',
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
