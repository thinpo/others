"""
ASX Market Data Processor
An integrated script to download, parse, and analyze ASX futures market data.
Combines functionality from asx_downloader, parse_asx_futures, and process_date_range.

Usage:
  python asx_processor.py --date YYMMDD [options]
  python asx_processor.py --date-range START_DATE END_DATE [options]
"""

import os
import sys
import time
import logging
import argparse
import re
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
    # Handle version compatibility
    try:
        from playwright._impl._api_types import Error as PlaywrightError
    except ImportError:
        from playwright.sync_api import Error as PlaywrightError
except ImportError:
    print("Playwright library not found. Installing required dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    print("Dependencies installed. Please run the script again.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded configuration
DEFAULT_CONFIG = {
    "output_dir": "downloads",
    "csv_dir": "csv",
    "timeout": 60000,  # milliseconds
    "delay": 5,        # seconds between downloads
    "wait_time": 60,   # seconds for CAPTCHA
    "headless": True,
    "browser": "chromium",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "viewport": {
        "width": 1920,
        "height": 1080
    },
    "retries": 3
}

# Report type configurations
REPORT_TYPES = {
    "market": {
        "name": "Market Summary Report",
        "url_pattern": "https://www.asx.com.au/data/futures/reports/EODWebMarketSummary{date}SFT.htm",
        "file_pattern": "asx_futures_{date}.html",
        "csv_pattern": "asx_futures_{date}.csv"
    },
    "offmarket": {
        "name": "Off Market Report",
        "url_pattern": "https://www.asx.com.au/data/futures/reports/DailyOffMarket{date}S.htm",
        "file_pattern": "asx_offmarket_{date}.html",
        "csv_pattern": "asx_offmarket_{date}.csv"
    }
}

# Australian holidays (format: MM-DD)
AUSTRALIAN_HOLIDAYS = [
    "01-01",  # New Year's Day
    "01-26",  # Australia Day
    "04-07",  # Good Friday (2023)
    "04-10",  # Easter Monday (2023)
    "04-25",  # Anzac Day
    "06-12",  # King's Birthday
    "12-25",  # Christmas Day
    "12-26",  # Boxing Day
]


class ASXProcessor:
    """Class to handle downloading, parsing, and analyzing ASX futures data."""
    
    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or DEFAULT_CONFIG
        
        # Ensure directories exist
        self.output_dir = self._ensure_dir_exists(self.config.get("output_dir"))
        self.csv_dir = self._ensure_dir_exists(self.config.get("csv_dir"))
    
    def _ensure_dir_exists(self, directory):
        """Create directory if it doesn't exist."""
        os.makedirs(directory, exist_ok=True)
        return directory
    
    def process_date_range(self, start_date, end_date, report_type="market", analyze=True):
        """Process a range of dates."""
        date_list = self._generate_date_range(start_date, end_date)
        logger.info(f"Processing {len(date_list)} dates from {start_date} to {end_date}")
        
        results = {
            "processed": 0,
            "download_success": 0,
            "download_failed": 0,
            "parse_success": 0,
            "parse_failed": 0,
            "analyze_success": 0,
            "analyze_failed": 0,
            "skipped_weekend": 0,
            "skipped_holiday": 0
        }
        
        for date_str in date_list:
            logger.info(f"Processing date: {date_str}")
            
            # Skip weekends
            if self._is_weekend(date_str):
                logger.info(f"Skipping weekend date: {date_str}")
                results["skipped_weekend"] += 1
                continue
            
            # Skip holidays
            if self._is_holiday(date_str):
                logger.info(f"Skipping holiday date: {date_str}")
                results["skipped_holiday"] += 1
                continue
                
            results["processed"] += 1
            
            # Download
            html_file = self.download_report(date_str, report_type)
            if html_file:
                results["download_success"] += 1
                
                # Parse
                csv_file = self.parse_html_to_csv(html_file, report_type)
                if csv_file:
                    results["parse_success"] += 1
                    
                    # Analyze if requested
                    if analyze and csv_file:
                        success = self.analyze_csv(csv_file)
                        if success:
                            results["analyze_success"] += 1
                        else:
                            results["analyze_failed"] += 1
                else:
                    results["parse_failed"] += 1
            else:
                results["download_failed"] += 1
                
            # Add delay between downloads
            if date_str != date_list[-1]:  # No delay after the last date
                time.sleep(self.config.get("delay", 5))
        
        # Print summary
        logger.info("=" * 50)
        logger.info(f"Processing Summary:")
        logger.info(f"Total dates in range: {len(date_list)}")
        logger.info(f"Dates processed: {results['processed']}")
        logger.info(f"Weekends skipped: {results['skipped_weekend']}")
        logger.info(f"Holidays skipped: {results['skipped_holiday']}")
        logger.info(f"Downloads: {results['download_success']} succeeded, {results['download_failed']} failed")
        logger.info(f"Parsing: {results['parse_success']} succeeded, {results['parse_failed']} failed")
        
        if analyze:
            logger.info(f"Analysis: {results['analyze_success']} succeeded, {results['analyze_failed']} failed")
            
        return results
    
    def download_report(self, date_str, report_type="market"):
        """
        Download ASX report for a specific date.
        
        Args:
            date_str: Date string in YYMMDD format
            report_type: Type of report (market or offmarket)
            
        Returns:
            str: Path to the downloaded file or None if download failed
        """
        # Get report configuration
        report_config = REPORT_TYPES.get(report_type)
        if not report_config:
            logger.error(f"Invalid report type: {report_type}")
            return None
        
        # Build URL and output filename
        url = report_config["url_pattern"].format(date=date_str)
        output_filename = report_config["file_pattern"].format(date=date_str)
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Check if file already exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logger.info(f"File already exists: {output_path}")
            return output_path
            
        logger.info(f"Downloading {report_config['name']} for {date_str}...")
        
        # Retry loop
        max_retries = self.config.get("retries", 3)
        for attempt in range(1, max_retries + 1):
            logger.info(f"Attempt {attempt}/{max_retries}...")
            
            try:
                with sync_playwright() as playwright:
                    # Select browser type
                    browser_factory = getattr(playwright, self.config.get("browser", "chromium"))
                    
                    # Launch browser
                    browser = browser_factory.launch(headless=self.config.get("headless", True))
                    context = browser.new_context(
                        viewport=self.config.get("viewport", {"width": 1920, "height": 1080}),
                        user_agent=self.config.get("user_agent")
                    )
                    
                    # Create a new page
                    page = context.new_page()
                    
                    # Set page timeout
                    page.set_default_timeout(self.config.get("timeout", 60000))
                    
                    # Navigate to URL
                    logger.info(f"Visiting URL: {url}")
                    response = page.goto(url)
                    
                    # Check HTTP status code
                    status = response.status if response else 0
                    if status >= 400:
                        logger.warning(f"HTTP error: {status}")
                        if status == 404:
                            logger.error(f"Page not found. No data available for {date_str}.")
                            return None
                        raise Exception(f"HTTP status code: {status}")
                    
                    # Wait for page to load
                    page.wait_for_load_state("networkidle")
                    
                    # Check for captcha or other verification
                    if "captcha" in page.content().lower() or "verification" in page.content().lower():
                        logger.warning("CAPTCHA detected!")
                        
                        # If in headless mode, switch to non-headless for CAPTCHA
                        if self.config.get("headless", True):
                            logger.info("Restarting in non-headless mode for CAPTCHA...")
                            browser.close()
                            
                            # Re-launch with headless=False
                            browser = browser_factory.launch(headless=False)
                            context = browser.new_context(
                                viewport=self.config.get("viewport", {"width": 1920, "height": 1080}),
                                user_agent=self.config.get("user_agent")
                            )
                            page = context.new_page()
                            page.goto(url)
                        
                        # Wait for manual CAPTCHA completion
                        logger.info(f"Please complete the CAPTCHA in the browser window...")
                        logger.info(f"Waiting up to {self.config.get('wait_time', 60)} seconds for completion...")
                        
                        # Wait for navigation or timeout
                        try:
                            page.wait_for_load_state("networkidle", timeout=self.config.get("wait_time", 60) * 1000)
                        except Exception as e:
                            logger.error(f"Timeout waiting for CAPTCHA: {str(e)}")
                            browser.close()
                            continue  # Try again
                    
                    # Get the HTML content
                    html_content = page.content()
                    
                    # Save HTML content to file
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    
                    # Verify file was saved with proper content
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        logger.info(f"✓ Download successful: {output_path}")
                        browser.close()
                        return output_path
                    else:
                        logger.error("Downloaded file is too small or empty")
                        browser.close()
                        continue  # Try again
                        
            except Exception as e:
                logger.error(f"Error during download: {str(e)}")
                
                # If last attempt, give up
                if attempt == max_retries:
                    logger.error(f"Failed to download after {max_retries} attempts.")
                    return None
        
        return None
    
    def parse_html_to_csv(self, html_file, report_type="market"):
        """
        Parse HTML file to CSV format.
        
        Args:
            html_file: Path to HTML file
            report_type: Type of report (market or offmarket)
            
        Returns:
            str: Path to the CSV file or None if parsing failed
        """
        logger.info(f"Parsing HTML file: {html_file}")
        
        try:
            # Determine output CSV file name
            basename = os.path.basename(html_file)
            date_match = re.search(r'_(2\d{5})\.html$', basename)
            
            if not date_match:
                logger.error(f"Could not extract date from filename: {basename}")
                return None
                
            date_str = date_match.group(1)
            output_filename = REPORT_TYPES[report_type]["csv_pattern"].format(date=date_str)
            output_path = os.path.join(self.csv_dir, output_filename)
            
            # Parse based on report type
            if report_type == "market":
                df = self._parse_market_html(html_file)
            else:
                df = self._parse_offmarket_html(html_file)
            
            if df is not None and not df.empty:
                # Save to CSV
                df.to_csv(output_path, index=False)
                logger.info(f"Saved parsed data to {output_path}")
                return output_path
            else:
                logger.warning(f"No data extracted from {html_file}")
                return None
        
        except Exception as e:
            logger.error(f"Error parsing HTML file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _parse_market_html(self, html_file):
        """Parse ASX futures market HTML file."""
        try:
            # Read the HTML file
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract the date from the title or header
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2})', html_content)
            report_date = date_match.group(1) if date_match else "Unknown"
            
            # Data structure to hold all contracts
            all_data = []
            
            # Find all contract tables (they have a header row with class="Headbold")
            contract_headers = soup.find_all('tr', class_='Headbold')
            
            for header in contract_headers:
                # Get contract name
                contract_name = header.get_text().strip()
                
                # Get the table that contains this header
                table = header.parent
                
                # Find the row with column headers (it's a row with class="noHighlight")
                column_header_row = table.find('tr', class_='noHighlight')
                if not column_header_row:
                    continue
                    
                # Extract column headers
                headers = []
                for td in column_header_row.find_all('td'):
                    headers.append(td.get_text().strip())
                
                # Find data rows (with class="Highlight")
                data_rows = table.find_all('tr', class_='Highlight')
                
                # Extract data from each row
                for row in data_rows:
                    row_data = {}
                    cells = row.find_all('td')
                    
                    # Make sure we have the right number of cells
                    if len(cells) != len(headers):
                        continue
                    
                    # Extract data into a dictionary
                    for i, cell in enumerate(cells):
                        row_data[headers[i]] = cell.get_text().strip()
                    
                    # Add contract name and report date
                    row_data['Contract'] = contract_name
                    row_data['Report Date'] = report_date
                    
                    # Add to our data collection
                    all_data.append(row_data)
            
            # Create DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                return df
            else:
                logger.warning("No data rows found in the HTML file")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing market HTML file: {str(e)}")
            return None
    
    def _parse_offmarket_html(self, html_file):
        """Parse ASX off-market HTML file."""
        try:
            # Read the HTML file
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract the date from the title or header
            date_match = re.search(r'trading date (\d{1,2}/\d{1,2}/\d{2})', html_content)
            report_date = date_match.group(1) if date_match else "Unknown"
            
            # Data structure to hold all contracts
            all_data = []
            
            # Find all section headers (tableHead)
            section_headers = soup.find_all('tr', class_='tableHead')
            
            # Process each section
            current_section = "Unknown"
            
            for section in section_headers:
                # Get section name
                section_name = section.get_text().strip()
                current_section = section_name
                
                # Find the next section header or end of document
                next_section = section.find_next('tr', class_='tableHead')
                
                # Get all Headbold rows between this section and the next
                current = section
                while True:
                    # Find next Headbold
                    current = current.find_next('tr', class_='Headbold')
                    
                    # Stop if we've reached the next section or end of document
                    if not current or (next_section and current.sourceline >= next_section.sourceline):
                        break
                    
                    # Process this contract
                    contract_name = current.get_text().strip()
                    
                    # Find the header row with column headers
                    header_row = current.find_next('tr', class_='noHighlight')
                    if not header_row:
                        continue
                    
                    # Extract headers
                    headers = []
                    for td in header_row.find_all('td'):
                        header_text = td.get_text().strip()
                        if header_text:
                            headers.append(header_text)
                    
                    # Skip the black line after the header
                    divider = header_row.find_next('tr')
                    
                    # Get all data rows until next black line or next contract
                    current_row = divider
                    while current_row:
                        current_row = current_row.find_next('tr')
                        
                        # Break if we've reached a divider, next contract or next section
                        if not current_row or \
                           (current_row.find('img') and 'black.gif' in current_row.find('img').get('src', '')) or \
                           ('Headbold' in current_row.get('class', [])) or \
                           ('tableHead' in current_row.get('class', [])):
                            break
                        
                        # Skip summary rows (those without the right classes)
                        if not current_row.get('class') or not ('Highlight' in current_row.get('class', []) or 'noHighlight' in current_row.get('class', [])):
                            continue
                        
                        # Extract data
                        cells = current_row.find_all('td')
                        if len(cells) < len(headers):
                            continue
                        
                        row_data = {}
                        for i, cell in enumerate(cells):
                            if i < len(headers):
                                row_data[headers[i]] = cell.get_text().strip()
                        
                        # Add metadata
                        row_data['Section'] = current_section
                        row_data['Contract'] = contract_name
                        row_data['Report Date'] = report_date
                        
                        # Add to our collection
                        all_data.append(row_data)
            
            # Create DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                return df
            else:
                # Check if there's a "No Data Available" message
                no_data_msg = soup.find(string=re.compile(r'No Data Available|No Trading Activity', re.IGNORECASE))
                if no_data_msg:
                    logger.info("Report indicates 'No Data Available' or 'No Trading Activity'")
                    # Create an empty DataFrame with just the report date
                    df = pd.DataFrame([{'Report Date': report_date, 'Message': 'No Data Available'}])
                    return df
                else:
                    logger.warning("No data rows found in the off-market HTML file")
                    return None
                
        except Exception as e:
            logger.error(f"Error parsing off-market HTML file: {str(e)}")
            return None
    
    def analyze_csv(self, csv_file):
        """
        Analyze a CSV file and print summary information.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            bool: True if analysis was successful
        """
        try:
            logger.info(f"Analyzing CSV file: {csv_file}")
            
            # Determine data type (market or offmarket) from filename
            is_offmarket = "offmarket" in os.path.basename(csv_file).lower()
            data_type = "Off-Market" if is_offmarket else "Market"
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Display basic information
            logger.info(f"CSV file: {csv_file} ({data_type} data)")
            logger.info(f"Number of rows: {len(df)}")
            logger.info(f"Number of columns: {len(df.columns)}")
            
            # Display unique contracts or sections
            if 'Contract' in df.columns:
                logger.info(f"Unique contracts: {df['Contract'].nunique()}")
            
            if 'Section' in df.columns:
                logger.info(f"Sections: {df['Section'].nunique()}")
            
            # Identify numeric columns based on data type
            if is_offmarket:
                potential_numeric = ['Price', 'Volume', 'Strike']
            else:
                potential_numeric = ['Open', 'High', 'Low', 'Last', 'Sett', 'Sett Chg', 'Op Int', 'Op Int Chg', 'Volume']
            
            # Filter to only include columns that actually exist in the dataframe
            numeric_columns = [col for col in potential_numeric if col in df.columns]
            
            # Convert to numeric, coercing errors to NaN
            for col in numeric_columns:
                if col in df.columns:
                    if df[col].dtype == 'object':  # Only convert string columns
                        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            
            # Summary for numeric columns
            if numeric_columns:
                for col in numeric_columns:
                    if col in df.columns:
                        logger.info(f"Total {col}: {df[col].sum():,.0f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing CSV: {str(e)}")
            return False
    
    def _generate_date_range(self, start_date, end_date):
        """Generate a list of dates in YYMMDD format."""
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, "%y%m%d")
        end_dt = datetime.strptime(end_date, "%y%m%d")
        
        # Generate dates
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime("%y%m%d"))
            current_dt += timedelta(days=1)
        
        return date_list
    
    def _is_weekend(self, date_str):
        """Check if the date is a weekend (Saturday or Sunday)."""
        dt = datetime.strptime(date_str, "%y%m%d")
        # Weekday returns 0-6 (Monday-Sunday), so 5 and 6 are weekend
        return dt.weekday() >= 5
    
    def _is_holiday(self, date_str):
        """Check if the date is an Australian holiday."""
        dt = datetime.strptime(date_str, "%y%m%d")
        mm_dd = dt.strftime("%m-%d")
        return mm_dd in AUSTRALIAN_HOLIDAYS


def validate_date_format(date_str):
    """Validate YYMMDD date format."""
    if not date_str or not re.match(r'^\d{6}$', date_str):
        return False
    
    try:
        datetime.strptime(date_str, '%y%m%d')
        return True
    except ValueError:
        return False


def main():
    """Main function to handle command-line arguments and run the processor."""
    parser = argparse.ArgumentParser(
        description="ASX Market Data Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Date arguments
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date", help="Single date (YYMMDD format, e.g., 240301)")
    date_group.add_argument("--date-range", nargs=2, metavar=("START_DATE", "END_DATE"), 
                           help="Date range (YYMMDD format, e.g., 240301 240310)")
    
    # Report type
    parser.add_argument("--report-type", choices=["market", "offmarket", "both"], default="market",
                      help="Report type to process")
    
    # Processing options
    parser.add_argument("--no-analyze", dest="analyze", action="store_false", default=True,
                      help="Skip analysis of CSV files")
    
    # Output directories
    parser.add_argument("--output-dir", default="downloads",
                      help="Directory for downloaded HTML files")
    parser.add_argument("--csv-dir", default="csv",
                      help="Directory for CSV files")
    
    # Headless mode
    parser.add_argument("--no-headless", dest="headless", action="store_false", default=True,
                      help="Don't use headless mode (show browser UI)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate dates
    if args.date and not validate_date_format(args.date):
        logger.error(f"Invalid date format: {args.date}. Use YYMMDD format (e.g., 240301)")
        sys.exit(1)
    
    if args.date_range:
        start_date, end_date = args.date_range
        if not validate_date_format(start_date) or not validate_date_format(end_date):
            logger.error(f"Invalid date range format. Use YYMMDD format (e.g., 240301)")
            sys.exit(1)
        
        # Check if start date is before end date
        start_dt = datetime.strptime(start_date, "%y%m%d")
        end_dt = datetime.strptime(end_date, "%y%m%d")
        if start_dt > end_dt:
            logger.error(f"Start date {start_date} is after end date {end_date}")
            sys.exit(1)
    
    # Create custom config with user options
    config = DEFAULT_CONFIG.copy()
    config["output_dir"] = args.output_dir
    config["csv_dir"] = args.csv_dir
    config["headless"] = args.headless
    
    # Initialize processor
    processor = ASXProcessor(config)
    
    # Process based on arguments
    if args.date:
        # Single date processing
        if args.report_type == "both":
            # Process market data
            logger.info(f"Processing market data for {args.date}")
            market_html = processor.download_report(args.date, "market")
            if market_html:
                market_csv = processor.parse_html_to_csv(market_html, "market")
                if market_csv and args.analyze:
                    processor.analyze_csv(market_csv)
            
            # Process offmarket data
            logger.info(f"Processing off-market data for {args.date}")
            offmarket_html = processor.download_report(args.date, "offmarket")
            if offmarket_html:
                offmarket_csv = processor.parse_html_to_csv(offmarket_html, "offmarket")
                if offmarket_csv and args.analyze:
                    processor.analyze_csv(offmarket_csv)
        else:
            # Process specific report type
            html_file = processor.download_report(args.date, args.report_type)
            if html_file:
                csv_file = processor.parse_html_to_csv(html_file, args.report_type)
                if csv_file and args.analyze:
                    processor.analyze_csv(csv_file)
    else:
        # Date range processing
        start_date, end_date = args.date_range
        
        if args.report_type == "both":
            # Process market data
            logger.info(f"Processing market data for date range {start_date} to {end_date}")
            processor.process_date_range(start_date, end_date, "market", args.analyze)
            
            # Process offmarket data
            logger.info(f"Processing off-market data for date range {start_date} to {end_date}")
            processor.process_date_range(start_date, end_date, "offmarket", args.analyze)
        else:
            # Process specific report type
            processor.process_date_range(start_date, end_date, args.report_type, args.analyze)


if __name__ == "__main__":
    main() 