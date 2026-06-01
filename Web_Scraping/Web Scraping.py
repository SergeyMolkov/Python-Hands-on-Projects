import sys
import subprocess

required_packages = {
    "selenium": "selenium",
    "webdriver-manager": "webdriver_manager"
}

for pip_name, import_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name]
        )

import os
import csv
import time
import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
       
def scrape_yelp_selenium():
    url = input("\nEnter Yelp restaurant URL: ").strip()
    
    if not url:
        print("Error: No URL entered!")
        return

    # Create folder
    save_folder = "Review_scraped"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created folder: {save_folder}/")

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        print(f"\nOpening page: {url}")
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        time.sleep(7)

        # Handling popups
        print("Handling popups...")
        try:
            close_buttons = driver.find_elements(By.CSS_SELECTOR, 
                "button[aria-label*='Close'], button[class*='close'], .modal-close")
            for btn in close_buttons[:3]:
                if btn.is_displayed():
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(2)
        except:
            pass

        # Restaurant name
        try:
            name = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1"))).text.strip()
        except:
            name = "Unknown_Restaurant"

        # Total reviews
        total_reviews = "Not found"
        try:
            review_elements = driver.find_elements(By.XPATH,
                "//span[contains(text(), 'review')] | //a[contains(@href, '#reviews')]")
            for elem in review_elements:
                text = elem.text.strip()
                if any(char.isdigit() for char in text):
                    total_reviews = text
                    break
        except:
            pass

        print(f"Restaurant: {name}")
        print(f"Total Reviews: {total_reviews}")

        # Scroll
        print("\nScrolling to load reviews...")
        for _ in range(12):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(random.uniform(2.5, 4.0))

        # Extract reviews
        review_cards = driver.find_elements(By.CSS_SELECTOR, 
            'li[class*="y-css"], div[class*="review"], article, div[data-testid*="review"]')

        print(f"Found {len(review_cards)} potential review blocks.")

        reviews = []
        for card in review_cards[:40]:
            try:
                reviewer = "Anonymous"
                for sel in ['div[class*="user-passport-info"] span', 'a[href*="/user_details"]', 'strong']:
                    elems = card.find_elements(By.CSS_SELECTOR, sel)
                    if elems:
                        text = elems[0].text.strip()
                        if text and len(text) > 1:
                            reviewer = text
                            break

                rating = "Not found"
                rating_elems = card.find_elements(By.CSS_SELECTOR, 'div[role="img"], span[aria-label*="star"]')
                if rating_elems:
                    rating = rating_elems[0].get_attribute("aria-label") or rating_elems[0].text

                try:
                    read_more = card.find_element(By.XPATH, ".//button[contains(text(), 'Read more')]")
                    driver.execute_script("arguments[0].click();", read_more)
                    time.sleep(0.7)
                except:
                    pass

                text_elems = card.find_elements(By.CSS_SELECTOR, 'p[lang], p[class*="comment"]')
                review_text = text_elems[0].text.strip() if text_elems else ""

                if review_text and len(review_text) > 40:
                    reviews.append({
                        "reviewer": reviewer,
                        "rating": rating,
                        "review_text": review_text
                    })
            except:
                continue

        print(f"Successfully extracted {len(reviews)} reviews.")

        # Save file with full path
        safe_name = "".join(c if c.isalnum() else "_" for c in name)[:50]
        csv_filename = f"{save_folder}/{safe_name}_reviews.csv"

        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Restaurant Name", name])
            writer.writerow(["Total Reviews", total_reviews])
            writer.writerow([])
            writer.writerow(["Reviewer", "Rating", "Review Text"])
            
            for review in reviews:
                writer.writerow([review["reviewer"], review["rating"], review["review_text"]])

        full_path = os.path.abspath(csv_filename)
        print(f"\n✅ File successfully saved!")
        print(f"   Full path: {full_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        driver.quit()
        print("Browser closed.")


if __name__ == "__main__":
    scrape_yelp_selenium()