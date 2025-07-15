#!/usr/bin/env python3
"""
Cookie-based authentication module for TDnet Search
Loads authentication cookies from tdnet_cookies.json
"""

import os
import json
import requests
from pathlib import Path

def load_tdnet_cookies():
    """
    Load TDnet Search cookies from tdnet_cookies.json file.
    
    Returns:
        dict: Dictionary of cookies or None if file not found
    """
    cookies_file = "tdnet_cookies.json"
    
    try:
        if os.path.exists(cookies_file):
            with open(cookies_file, 'r', encoding='utf-8') as f:
                cookies_data = json.load(f)
            
            print(f"✓ Loaded cookies from {cookies_file}")
            
            # Handle both single cookie object and array of cookies
            if isinstance(cookies_data, dict):
                cookies_data = [cookies_data]
            elif not isinstance(cookies_data, list):
                print("❌ Invalid cookies format - expected dict or list")
                return None
            
            # Convert to simple name-value pairs
            cookies = {}
            for cookie in cookies_data:
                if isinstance(cookie, dict) and 'name' in cookie and 'value' in cookie:
                    cookies[cookie['name']] = cookie['value']
            
            if cookies:
                print(f"✓ Processed {len(cookies)} cookies")
                # Look for important authentication cookies
                auth_cookies = ['SACSID', 'ACSID', '__Secure-1PSID', '__Secure-3PSID']
                found_auth = [name for name in auth_cookies if name in cookies]
                if found_auth:
                    print(f"✓ Found authentication cookies: {found_auth}")
                else:
                    print("⚠️  No standard authentication cookies found")
                
                return cookies
            else:
                print("❌ No valid cookies found in file")
                return None
        else:
            print(f"❌ Cookies file not found: {cookies_file}")
            print("Please create tdnet_cookies.json with your authentication cookies")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in cookies file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading cookies: {e}")
        return None

def create_authenticated_session():
    """
    Create an authenticated requests session using cookies from tdnet_cookies.json.
    
    Returns:
        requests.Session: Authenticated session or None if failed
    """
    print("=" * 60)
    print("CREATING AUTHENTICATED SESSION WITH COOKIES")
    print("=" * 60)
    
    # Load cookies
    cookies = load_tdnet_cookies()
    if not cookies:
        return None
    
    # Create session
    session = requests.Session()
    
    # Set headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'https://tdnet-search.appspot.com/'
    })
    
    # Add all cookies to the session
    for name, value in cookies.items():
        session.cookies.set(
            name=name,
            value=value,
            domain='.tdnet-search.appspot.com',
            path='/'
        )
    
    print(f"✓ Created session with {len(cookies)} cookies")
    
    # Test the session by making a request to TDnet Search
    try:
        print("Testing authenticated session...")
        response = session.get("https://tdnet-search.appspot.com/", timeout=10)
        response.raise_for_status()
        
        print(f"✓ Session test successful (status: {response.status_code})")
        
        # Check if we appear to be logged in
        if "Logout" in response.text:
            print("✓ Authentication confirmed - logout link found")
        else:
            print("⚠️  Warning: No logout link found - authentication may not be working")
        
        return session
        
    except Exception as e:
        print(f"❌ Session test failed: {e}")
        return None

def test_authentication():
    """
    Test function to verify authentication is working.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    session = create_authenticated_session()
    if session:
        print("🎉 Authentication test passed!")
        return True
    else:
        print("💥 Authentication test failed!")
        return False

# Convenience function for backward compatibility
def get_authenticated_session(headless=True):
    """
    Get an authenticated requests session for TDnet Search.
    
    Args:
        headless (bool): Ignored (kept for compatibility)
        
    Returns:
        tuple: (session, None) - auth handler is None since we don't use Selenium
    """
    session = create_authenticated_session()
    return session, None

if __name__ == "__main__":
    # Test the authentication when run directly
    test_authentication() 