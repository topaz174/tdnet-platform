#!/usr/bin/env python3
"""
Test script for XBRL parser - runs financial table generator
for a diverse set of 20 Japanese companies.
Assumes earnings data has already been scraped and stored in the database.
"""

import subprocess
import sys
import os
from pathlib import Path

# List of 20 diverse Japanese companies across different sectors
TEST_COMPANIES = [
    "1023",
    "1232",
    "1253",
    "1385",
    "1448",
    "1514",
    "1583",
    "1639",
    "1669",
    "1685",
    "1748",
    "1776",
    "2231",
    "2286",
    "2389",
    "2413",
    "2425",
    "2616",
    "2625",
    "2707",
    "2716",
    "2934",
    "3001",
    "3013",
    "3230",
    "3312",
    "3387",
    "3469",
    "3602",
    "3688",
    "3695",
    "3805",
    "3836",
    "3864",
    "3876",
    "3915",
    "3933",
    "4017",
    "4038",
    "4121",
    "4139",
    "4248",
    "4334",
    "4434",
    "4622",
    "4674",
    "4682",
    "4733",
    "4831",
    "4841",
    "4892",
    "4927",
    "4945",
    "5030",
    "5078",
    "5260",
    "5328",
    "5388",
    "5409",
    "5427",
    "5451",
    "5495",
    "5523",
    "5555",
    "5641",
    "5643",
    "5851",
    "5871",
    "5975",
    "5980",
    "5997",
    "6020",
    "6091",
    "6281",
    "6426",
    "6446",
    "6466",
    "6707",
    "6801",
    "6808",
    "6812",
    "6825",
    "6876",
    "7134",
    "7152",
    "7218",
    "7224",
    "7373",
    "7423",
    "7447",
    "7465",
    "7479",
    "7623",
    "8259",
    "8401",
    "8453",
    "8454",
    "8459",
    "8619",
    "9052",
    "9108",
    "9116",
    "9153",
    "9201",
    "9267",
    "9295",
    "9446",
    "9482",
    "9507",
    "9589",
    "9677",
    "9693",
    "9707",
    "9718",
    "9735",
    "9758",
    "9813",
    "9814",
    "9939",
    "9966"
]


def run_financial_table_generator(company_code):
    """Run the financial table generator for a given company code."""
    print(f"\n{'='*60}")
    print(f"Running financial table generator for company: {company_code}")
    print(f"{'='*60}")
    
    try:
        # Run the financial table generator
        cmd = [
            sys.executable, "-X", "utf8",  # Force UTF-8 mode for the subprocess
            "src/xbrl_parser/generate_financial_table.py"
        ]

        # Ensure UTF-8 for all stdio in the child process
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            cmd,
            input=company_code,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        
        if result.returncode == 0:
            # Check if the output indicates no data was found
            stdout_text = result.stdout.lower()
            if "no 'earnings report' disclosures found" in stdout_text or "no data processed" in stdout_text:
                print(f"[NOT FOUND] No earnings data found for company {company_code}")
                return "not_found"
            else:
                print(f"[SUCCESS] Financial table generator completed for {company_code}")
                return True
        else:
            print(f"[ERROR] Financial table generator failed for {company_code}")
            # Only print stderr if it's not a Unicode issue
            try:
                print(f"Error: {result.stderr}")
            except:
                print("Error: (encoding issue in error message)")
            return False
            
    except Exception as e:
        print(f"[EXCEPTION] Error running financial table generator for {company_code}: {e}")
        return False

def main():
    """Main function to test XBRL financial table generator with multiple companies."""
    print("XBRL Financial Table Generator Test Suite")
    print("=" * 80)
    print(f"Generating Excel tables for {len(TEST_COMPANIES)} diverse Japanese companies")
    print("Note: Assumes earnings data has already been scraped and stored in database")
    print("=" * 80)
    
    successful_companies = []
    failed_companies = []
    not_found_companies = []
    
    for i, company_code in enumerate(TEST_COMPANIES, 1):
        print(f"\n\nProcessing company {i}/{len(TEST_COMPANIES)}: {company_code}")
        
        try:
            # Run financial table generator
            result = run_financial_table_generator(company_code)
            
            if result is True:
                successful_companies.append(company_code)
                print(f"[COMPLETE] Company {company_code} processed successfully")
            elif result == "not_found":
                not_found_companies.append(company_code)
                print(f"[NO DATA] Company {company_code} - no earnings data in database")
            else:
                failed_companies.append(company_code)
                print(f"[FAILED] Company {company_code} processing failed")
            
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Test interrupted by user at company {company_code}")
            break
        except Exception as e:
            print(f"[FAILED] Failed to process company {company_code}: {e}")
            failed_companies.append(company_code)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    total_tested = len(successful_companies) + len(failed_companies) + len(not_found_companies)
    print(f"Total companies tested: {total_tested}")
    print(f"Successful: {len(successful_companies)}")
    print(f"No data found: {len(not_found_companies)}")
    print(f"Failed: {len(failed_companies)}")
    
    if successful_companies:
        print(f"\n[SUCCESS] Successful companies: {', '.join(successful_companies)}")
    
    if not_found_companies:
        print(f"\n[NO DATA] Companies with no earnings data: {', '.join(not_found_companies)}")
    
    if failed_companies:
        print(f"\n[FAILED] Failed companies: {', '.join(failed_companies)}")
    
    print(f"\nCheck 'xbrl_tables/' directory for generated Excel files")
    print(f"Files will be named: [company_code]_[company_name]_financial_table.xlsx")

if __name__ == "__main__":
    main() 