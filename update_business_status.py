from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--start-maximized')
# chrome_options.add_argument('--headless')  # Uncomment to run in headless mode

# Initialize Selenium Wire with Chrome driver
driver = webdriver.Chrome(options=chrome_options)

try:
    # Open the URL
    print("Opening https://matchapro.web.bps.go.id/dirgc...")
    driver.get('https://matchapro.web.bps.go.id/dirgc')
    
    print("Page loaded successfully!")
    
    # Optional: Inspect requests and responses
    print("\n--- Captured Requests ---")
    for request in driver.requests:
        print(f'Request: {request.method} {request.url}')
        if request.response:
            print(f'Response: {request.response.status_code}')
    
    # Keep the browser open for 15 seconds to see the page
    print("\nBrowser will close in 15 seconds...")
    time.sleep(15)

except Exception as e:
    print(f"Error occurred: {str(e)}")

finally:
    # Close the browser
    driver.quit()
    print("Browser closed.")
