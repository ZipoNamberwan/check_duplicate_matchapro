from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time
from dotenv import load_dotenv, dotenv_values



def getCredentialsFromEnv():
    # Load environment variables from .env file
    load_dotenv()
    env_vars = dotenv_values(".env")

    """Get username and password from .env file"""
    username = env_vars.get('username')
    password = env_vars.get('password')
    if not username or not password:
        print("ERROR: username or password not found in .env file!")
        return None, None
    return username, password

def click_button_by_xpath(driver, xpath):
    """
    Find and click a button using the provided XPath.
    
    Args:
        driver: Selenium WebDriver instance
        xpath: The XPath of the element to click
    """
    try:
        # Wait for the element to be clickable (max 10 seconds)
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        print(f"Button clicked successfully!")
        time.sleep(2)  # Wait for page to respond
    except Exception as e:
        print(f"Error clicking button: {str(e)}")

def fillAndSubmitLoginForm(driver):
    """Fill the login form and submit it"""
    try:
        # Wait for the login form to appear
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/div/div/div/div/form/div[1]/input")))
        print("Login form loaded!")
        
        # Get credentials from .env
        username, password = getCredentialsFromEnv()
        if not username or not password:
            return False
        
        # Fill username field
        username_field = driver.find_element(By.XPATH, "/html/body/div/div[2]/div/div/div/div/form/div[1]/input")
        username_field.clear()
        username_field.send_keys(username)
        print(f"Username filled: {username}")
        
        # Fill password field
        password_field = driver.find_element(By.XPATH, "/html/body/div/div[2]/div/div/div/div/form/div[2]/input")
        password_field.clear()
        password_field.send_keys(password)
        print("Password filled!")
        
        # Submit the login form
        submit_button = driver.find_element(By.XPATH, "/html/body/div/div[2]/div/div/div/div/form/div[4]/input[2]")
        submit_button.click()
        print("Login form submitted!")
        return True
        
    except TimeoutException:
        print("ERROR: Login form did not appear within 10 seconds!")
        return False
    except NoSuchElementException:
        print("ERROR: One or more login form elements not found!")
        return False
    except Exception as e:
        print(f"ERROR: Failed to fill and submit login form - {str(e)}")
        return False

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
    
    # Click the button using the provided XPath
    button_xpath = '/html/body/div[2]/div[3]/div/div/div/div/div/form/a'
    click_button_by_xpath(driver, button_xpath)

    # Fill and submit the login form
    fillAndSubmitLoginForm(driver)
    
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
