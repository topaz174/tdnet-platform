import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

def scrape_tdnet_titles(date_str, start_hour=None, end_hour=None):
    """
    Scrape TDnet disclosure titles for a given date and time range.
    
    Args:
        date_str (str): Date in format 'MM/DD' or 'MM/DD/YYYY'.
        start_hour (int, optional): Start hour in 24-hour format. Default is None (beginning of day).
        end_hour (int, optional): End hour in 24-hour format. Default is None (end of day).
    
    Returns:
        list: List of disclosure titles
    """
    
    # Parse date
    date_parts = date_str.strip().split('/')
    if len(date_parts) == 2:
        # MM/DD, use current year
        year = datetime.now().year
        month, day = map(int, date_parts)
    elif len(date_parts) == 3:
        # MM/DD/YYYY
        month, day, year = map(int, date_parts)
    else:
        raise ValueError("Date must be in MM/DD or MM/DD/YYYY format.")
    
    date_for_url = f"{year}{month:02d}{day:02d}"
    
    base_url = "https://www.release.tdnet.info/inbs/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    all_titles = []
    page = 1
    
    # Convert hour integers to time objects for comparison
    start_time_obj = None
    end_time_obj = None
    
    if start_hour is not None:
        start_time_obj = datetime.now().replace(hour=start_hour, minute=0, second=0).time()
    
    if end_hour is not None:
        end_time_obj = datetime.now().replace(hour=end_hour, minute=0, second=0).time()
    
    while True:
        page_url = f"{base_url}I_list_{page:03d}_{date_for_url}.html"
        print(f"Scraping page {page} from: {page_url}")
        
        try:
            response = requests.get(page_url, headers=headers)
            
            if response.status_code == 404:
                print(f"Page {page} not found. Reached the end of disclosures for {date_str}.")
                break
                
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            main_table = soup.find('table', id='main-list-table')
            
            if not main_table:
                print(f"No disclosure table found on page {page} for {date_str}.")
                break
            
            rows = main_table.find_all('tr')
            
            if not rows:
                print(f"No disclosure rows found on page {page} for {date_str}.")
                break
            
            print(f"Found {len(rows)} disclosure rows on page {page} for {date_str}")
            
            # Process each row
            for row in rows:
                cells = row.find_all('td')
                
                if len(cells) < 7:  
                    continue
                
                disclosure_time = cells[0].get_text(strip=True)
                
                # Skip if not in time range
                if start_time_obj or end_time_obj:
                    hour, minute = map(int, disclosure_time.split(':'))
                    disclosure_time_obj = datetime.now().replace(hour=hour, minute=minute, second=0).time()
                    
                    if start_time_obj and disclosure_time_obj < start_time_obj:
                        continue
                    
                    if end_time_obj and disclosure_time_obj > end_time_obj:
                        continue
                
                title_cell = cells[3]
                link_tag = title_cell.find('a')
                
                if not link_tag:
                    continue  
                
                title = link_tag.get_text(strip=True)
                all_titles.append(title)
                
            page += 1
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Page {page} not found. Reached the end of disclosures for {date_str}.")
                break
            else:
                print(f"HTTP error scraping page {page} for {date_str}: {e}")
                break
        except Exception as e:
            print(f"Error scraping page {page} for {date_str}: {e}")
            break
    
    print(f"Total titles scraped: {len(all_titles)}")
    return all_titles

def save_titles_to_csv(titles, output_file="input.md"):
    """
    Save titles to a CSV file.
    
    Args:
        titles (list): List of titles to save
        output_file (str): Output file path
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for title in titles:
            writer.writerow([title])
    
    print(f"Saved {len(titles)} titles to {output_file}")

if __name__ == "__main__":
    date_str = input("Enter the date (MM/DD or MM/DD/YYYY): ")
    start_hour = input("Enter the start hour (0-23, optional): ")
    end_hour = input("Enter the end hour (0-23, optional): ")
    
    start_hour = int(start_hour) if start_hour else None
    end_hour = int(end_hour) if end_hour else None
    
    print(f"Scraping TDnet titles for {date_str}" + 
          (f" from {start_hour}:00" if start_hour is not None else "") + 
          (f" to {end_hour}:00" if end_hour is not None else ""))
    
    titles = scrape_tdnet_titles(date_str, start_hour, end_hour)
    save_titles_to_csv(titles) 