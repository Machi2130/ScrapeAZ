
import asyncio
import pandas as pd
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import re
import html
import time
from urllib.parse import urljoin
import random
from tabulate import tabulate
import json
from datetime import datetime
import logging
import heapq
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Set
import bisect
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from flask import send_file


app = Flask(__name__)
CORS(app)

# Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

class ProductTrie:
    """Trie data structure for efficient category classification (brand-agnostic)"""
    
    def __init__(self):
        self.root = {}
    
    def insert(self, word: str, category: str):
        """Insert word into trie with O(m) time complexity where m = word length"""
        node = self.root
        for char in word.lower():
            if char not in node:
                node[char] = {}
            node = node[char]
        
        node['is_end'] = True
        node['category'] = category
    
    def search(self, text: str) -> Optional[Dict]:
        """Search for categories in text with O(n*m) complexity"""
        text = text.lower()
        results = []
        
        for i in range(len(text)):
            node = self.root
            j = i
            current_word = ""
            
            while j < len(text) and text[j] in node:
                current_word += text[j]
                node = node[text[j]]
                
                if 'is_end' in node:
                    results.append({
                        'word': current_word,
                        'category': node.get('category'),
                        'position': i
                    })
                j += 1
        
        return results[0] if results else None

class ProductCache:
    """LRU Cache implementation using hash map + deque"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_order = deque()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Dict]:
        """Get item from cache - O(1) average case"""
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Dict):
        """Put item in cache - O(1) average case"""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            lru_key = self.access_order.popleft()
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)

class DuplicateTracker:
    """Set-based duplicate detection for O(1) lookups"""
    
    def __init__(self):
        self.seen_asins: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.seen_titles: Set[str] = set()
    
    def is_duplicate(self, product: Dict) -> bool:
        """Check if product is duplicate - O(1) time complexity"""
        asin = product.get('ASIN', '').strip()
        url = product.get('Product_URL', '').strip()
        title = product.get('Product_Name', '').strip().lower()
        
        if asin and asin in self.seen_asins:
            return True
        
        if url and url in self.seen_urls:
            return True
        
        if title and title in self.seen_titles:
            return True
        
        if asin:
            self.seen_asins.add(asin)
        if url:
            self.seen_urls.add(url)
        if title:
            self.seen_titles.add(title)
        
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get duplicate tracking statistics"""
        return {
            'unique_asins': len(self.seen_asins),
            'unique_urls': len(self.seen_urls),
            'unique_titles': len(self.seen_titles)
        }

class AdvancedAmazonScraper:
    """Enhanced Amazon scraper with model number extraction and brand-agnostic approach"""
    
    def __init__(self):
        self.products_data = []
        self.base_url = "https://www.amazon.in"
        self.category_trie = ProductTrie()
        self.product_cache = ProductCache(max_size=2000)
        self.duplicate_tracker = DuplicateTracker()
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'duplicates_found': 0,
            'processing_time': 0,
            'model_numbers_found': 0
        }
        self.setup_logging()
        self.initialize_category_trie()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_amazon_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_category_trie(self):
        """Initialize trie with category keywords (brand-agnostic)"""
        category_keywords = [
            ('phone', 'mobile'), ('smartphone', 'mobile'), ('mobile', 'mobile'), ('iphone', 'mobile'), ('android', 'mobile'),
            ('laptop', 'laptop'), ('notebook', 'laptop'), ('macbook', 'laptop'), ('chromebook', 'laptop'), ('ultrabook', 'laptop'),
            ('television', 'tv'), ('smart tv', 'tv'), ('led tv', 'tv'), ('oled', 'tv'), ('qled', 'tv'),
            ('headphones', 'audio'), ('earphones', 'audio'), ('earbuds', 'audio'), ('speaker', 'audio'), ('soundbar', 'audio'), ('airpods', 'audio'),
            ('watch', 'watch'), ('smartwatch', 'watch'), ('fitness tracker', 'watch'),
            ('power bank', 'powerbank'), ('powerbank', 'powerbank'), ('portable charger', 'powerbank'),
            ('camera', 'camera'), ('dslr', 'camera'), ('mirrorless', 'camera'),
            ('gaming', 'gaming'), ('console', 'gaming'), ('controller', 'gaming')
        ]
        
        for keyword, category in category_keywords:
            self.category_trie.insert(keyword, category)
        
        self.logger.info(f"Initialized category trie with {len(category_keywords)} category keywords")
    
    def detect_category(self, title: str) -> str:
        """Fast category detection using Trie - O(n*m) complexity"""
        result = self.category_trie.search(title)
        if result:
            return result['category']
        title_lower = title.lower()
        if any(word in title_lower for word in ['tablet', 'ipad']):
            return 'tablet'
        elif any(word in title_lower for word in ['router', 'wifi', 'modem']):
            return 'networking'
        elif any(word in title_lower for word in ['keyboard', 'mouse', 'webcam']):
            return 'accessories'
        elif any(word in title_lower for word in ['charger', 'cable', 'adapter']):
            return 'accessories'
        return 'general'
    
    def extract_brand_from_title(self, title: str) -> str:
        """Extract brand name from title using common patterns"""
        title = self.clean_text(title)  # Ensure title is cleaned
        title_words = title.split()
        if title_words:
            potential_brand = title_words[0]
            if potential_brand.lower() in ['new', 'latest', 'original', 'genuine'] and len(title_words) > 1:
                potential_brand = title_words[1]
            return potential_brand.strip()
        return 'Unknown'
    
    def extract_model_number_from_title(self, title: str) -> Optional[str]:
        """Extract model number from product title using regex patterns"""
        title = self.clean_text(title)  # Ensure title is cleaned
        patterns = [
            r'\b([A-Z]{1,4}[-]?[0-9]{2,6}[A-Z]{0,4})\b',
            r'\b([0-9]{2,4}[A-Z]{1,4}[0-9]{1,4})\b',
            r'\b(Model[:\s]+([A-Z0-9-]{4,}))\b',
            r'\b([A-Z]{2,4}[0-9]{3,6})\b',
            r'\b([0-9]{4}[A-Z]{2,4})\b'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, title, re.IGNORECASE)
            for match in matches:
                model = match.group(1) if '(' in pattern and 'Model' in pattern else match.group(0)
                if not re.match(r'^(GB|TB|MP|Hz|MHz|GHz|RAM|ROM|USB)$', model, re.IGNORECASE):
                    return model.strip()
        return None
    
    async def extract_model_number_from_product_page(self, product_url: str, crawler) -> Optional[str]:
        """Extract model number from product detail page with debug logging"""
        if not product_url or product_url == 'N/A':
            return None
        
        try:
            cache_key = f"model_{product_url}"
            cached_model = self.product_cache.get(cache_key)
            if cached_model:
                return cached_model.get('model_number')
            
            self.logger.info(f"Fetching model number from: {product_url[:100]}...")
            result = await crawler.arun(
                url=product_url,
                word_count_threshold=10,
                bypass_cache=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                delay_before_return_html=random.uniform(1, 2)
            )
            
            if not result.success:
                self.logger.warning(f"Failed to fetch product page: {result.error_message}")
                return None
            
            soup = BeautifulSoup(result.html, 'html.parser')
            model_selectors = [
                '#productDetails_detailBullets_sections1 tr',
                '#productDetails_techSpec_section_1 tr',
                '#productDetails_feature_div tr',
                '.prodDetTable tr',
                '#feature-bullets li',
                '.a-unordered-list.a-vertical li',
                '#technicalSpecifications_section_1 tr',
                '.comparison_table tr'
            ]
            
            for selector in model_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = self.clean_text(element.get_text()).lower()
                    if any(keyword in text for keyword in ['model number', 'item model number', 'model name', 'model']):
                        cells = element.find_all(['td', 'span'])
                        for i, cell in enumerate(cells):
                            cell_text = self.clean_text(cell.get_text().strip()).lower()
                            if any(keyword in cell_text for keyword in ['model number', 'item model number', 'model name', 'model']):
                                if i + 1 < len(cells):
                                    model_value = self.clean_text(cells[i + 1].get_text().strip())
                                    if model_value and len(model_value) > 2:
                                        self.logger.debug(f"Found model number: {model_value} (raw: {repr(cells[i + 1].get_text())})")
                                        self.product_cache.put(cache_key, {'model_number': model_value})
                                        self.metrics['model_numbers_found'] += 1
                                        return model_value
                                    else:
                                        self.logger.debug(f"Invalid model number: {repr(cells[i + 1].get_text())}")
            
            json_scripts = soup.find_all('script', {'type': 'application/ld+json'})
            for script in json_scripts:
                try:
                    data = json.loads(self.clean_text(script.string))
                    if isinstance(data, dict) and 'model' in data:
                        model_value = self.clean_text(data['model'])
                        if model_value and len(model_value) > 2:
                            self.logger.debug(f"Found model number in JSON-LD: {model_value}")
                            self.product_cache.put(cache_key, {'model_number': model_value})
                            self.metrics['model_numbers_found'] += 1
                            return model_value
                except:
                    continue
            
            meta_selectors = ['meta[name*="model"]', 'meta[property*="model"]']
            for selector in meta_selectors:
                meta_tag = soup.select_one(selector)
                if meta_tag and meta_tag.get('content'):
                    model_value = self.clean_text(meta_tag.get('content').strip())
                    if model_value and len(model_value) > 2:
                        self.logger.debug(f"Found model number in meta: {model_value}")
                        self.product_cache.put(cache_key, {'model_number': model_value})
                        self.metrics['model_numbers_found'] += 1
                        return model_value
            
            self.logger.debug(f"No model number found for {product_url[:100]}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting model number from product page: {str(e)}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with emoji removal"""
        if not text:
            return ""
        text = html.unescape(text)
        # Remove invisible Unicode characters
        text = re.sub(r'[\u200e\u200f\u202a-\u202e\u2060\ufeff\u200c\u200d‎‏\u00a0]', '', text)
        # Remove emojis
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_numeric_price(self, price_str: str) -> float:
        """Extract numeric price for mathematical operations"""
        price_str = self.clean_text(price_str)  # Ensure price is cleaned
        if not price_str or price_str == 'N/A':
            return 0.0
        cleaned = re.sub(r'[₹,]', '', price_str)
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    
    def extract_specifications(self, title: str, category: str) -> Dict[str, str]:
        """Enhanced specification extraction with category-specific patterns"""
        title = self.clean_text(title)  # Ensure title is cleaned
        specs = {}
        title_upper = title.upper()
        patterns = {
            'mobile': {
                'Storage': r'(\d+)\s*GB(?:\s+(?:Storage|ROM|Internal))?',
                'RAM': r'(\d+)\s*GB\s+RAM',
                'Camera': r'(\d+)\s*MP(?:\s+Camera)?',
                'Battery': r'(\d+)\s*mAh',
                'Display': r'(\d+\.?\d*)\s*(?:inch|")'
            },
            'laptop': {
                'Processor': r'(Intel\s+Core\s+i[3579]|AMD\s+Ryzen\s+[357]|Apple\s+M[12])',
                'RAM': r'(\d+)\s*GB(?:\s+RAM|DDR)',
                'Storage': r'(\d+(?:GB|TB))\s*(?:SSD|HDD)',
                'Display': r'(\d+\.?\d*)\s*(?:inch|")',
                'Graphics': r'(GTX|RTX|Radeon|Intel\s+Graphics)'
            },
            'tv': {
                'Screen_Size': r'(\d+)\s*(?:inch|")',
                'Resolution': r'(4K|UHD|Ultra\s*HD|Full\s*HD|FHD|HD)',
                'Smart_TV': r'(Smart\s*TV|Android\s*TV|WebOS)',
                'HDR': r'(HDR|HDR10|Dolby\s*Vision)'
            },
            'powerbank': {
                'Capacity': r'(\d+(?:,\d+)?)\s*mAh',
                'Charging_Power': r'(\d+)\s*W(?:\s+(?:Fast|Quick|Rapid))?',
                'Ports': r'(\d+)\s*(?:Port|USB)',
                'Type': r'(Type-C|USB-C|Micro\s*USB)'
            },
            'audio': {
                'Driver_Size': r'(\d+)\s*mm(?:\s+Driver)?',
                'Frequency': r'(\d+)\s*Hz(?:\s*-\s*\d+\s*kHz)?',
                'Battery': r'(\d+)\s*(?:hours?|hrs?)',
                'Connectivity': r'(Bluetooth\s*[\d.]+|Wireless|Wired)'
            }
        }
        
        category_patterns = patterns.get(category, {})
        for spec_name, pattern in category_patterns.items():
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                specs[spec_name] = match.group(1)
        return specs
    
    def extract_price_info(self, container) -> Dict[str, str]:
        """Enhanced price extraction with fallback mechanisms"""
        price_data = {'price': 'N/A', 'original_price': 'N/A', 'discount': 'N/A'}
        price_selectors = [
            '.a-price .a-offscreen', '.a-price-whole', '[data-a-color="price"] .a-offscreen',
            '.a-price-symbol + .a-price-whole', '.a-price', '.a-size-medium.a-color-price',
            'span.a-price-symbol + span'
        ]
        
        for selector in price_selectors:
            price_elem = container.select_one(selector)
            if price_elem:
                price_text = self.clean_text(price_elem.get_text())
                price_match = re.search(r'₹([\d,]+(?:\.\d{2})?)', price_text)
                if price_match:
                    price_data['price'] = f"₹{price_match.group(1)}"
                    break
        
        original_selectors = [
            '.a-price.a-text-price .a-offscreen', '.a-price-was .a-offscreen',
            '[data-a-strike="true"] .a-offscreen', '.a-text-strike'
        ]
        
        for selector in original_selectors:
            orig_elem = container.select_one(selector)
            if orig_elem:
                orig_text = self.clean_text(orig_elem.get_text())
                orig_match = re.search(r'₹([\d,]+(?:\.\d{2})?)', orig_text)
                if orig_match:
                    price_data['original_price'] = f"₹{orig_match.group(1)}"
                    break
        
        discount_selectors = ['.a-letter-space', '[data-a-color="price"]', '.savingsPercentage', '.a-size-large.a-color-price']
        for selector in discount_selectors:
            disc_elem = container.select_one(selector)
            if disc_elem:
                disc_text = self.clean_text(disc_elem.get_text())
                disc_match = re.search(r'(\d+)%', disc_text)
                if disc_match:
                    price_data['discount'] = f"{disc_match.group(1)}%"
                    break
        
        return price_data
    
    def extract_rating_info(self, container) -> Dict[str, str]:
        """Enhanced rating and review extraction"""
        rating_data = {'rating': 'N/A', 'review_count': 'N/A'}
        rating_selectors = ['.a-icon-alt', 'i.a-icon-star-small .a-icon-alt', '[aria-label*="stars"]', '.a-star-medium .a-icon-alt']
        
        for selector in rating_selectors:
            rating_elem = container.select_one(selector)
            if rating_elem:
                rating_text = self.clean_text(rating_elem.get('alt', '') or rating_elem.get('aria-label', '') or rating_elem.get_text())
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    rating_data['rating'] = f"{rating_match.group(1)}/5"
                    break
        
        review_selectors = ['.a-size-base', 'a[href*="reviews"]', '.a-link-normal[href*="reviews"]', '[data-csa-c-type="link"][href*="reviews"]']
        for selector in review_selectors:
            review_elem = container.select_one(selector)
            if review_elem:
                review_text = self.clean_text(review_elem.get_text())
                review_match = re.search(r'([\d,]+)', review_text)
                if review_match:
                    rating_data['review_count'] = review_match.group(1)
                    break
        
        return rating_data
    
    async def extract_product_info(self, container, crawler) -> Optional[Dict]:
        """Enhanced product information extraction with model number"""
        try:
            asin = container.get('data-asin', '')
            if asin:
                cached_product = self.product_cache.get(asin)
                if cached_product:
                    self.metrics['cache_hits'] += 1
                    return cached_product
            
            self.metrics['cache_misses'] += 1
            title_selectors = [
                'h2 a span', '.a-size-base-plus', '.a-size-medium', 'h2 a',
                '.s-size-mini span', 'a[data-csa-c-type="link"] span', '.a-text-normal', '.a-size-mini span'
            ]
            
            title = ""
            for selector in title_selectors:
                title_elem = container.select_one(selector)
                if title_elem:
                    title = self.clean_text(title_elem.get_text())
                    if title and len(title) > 10:
                        break
            
            if not title:
                return None
            
            link_selectors = ['h2 a', 'a[data-csa-c-type="link"]', 'a.a-link-normal']
            product_url = ""
            for selector in link_selectors:
                link_elem = container.select_one(selector)
                if link_elem and link_elem.get('href'):
                    product_url = urljoin(self.base_url, link_elem['href'])
                    break
            
            category = self.detect_category(title)
            brand = self.extract_brand_from_title(title)
            model_number = self.extract_model_number_from_title(title)
            
            if not model_number and product_url:
                model_number = await self.extract_model_number_from_product_page(product_url, crawler)
            
            specifications = self.extract_specifications(title, category)
            price_info = self.extract_price_info(container)
            rating_info = self.extract_rating_info(container)
            
            product_data = {
                'Product_Name': title,
                'Brand': brand,
                'Model_Number': model_number or 'N/A',
                'Category': category.title(),
                'Price': price_info['price'],
                'Original_Price': price_info['original_price'],
                'Discount': price_info['discount'],
                'Rating': rating_info['rating'],
                'Review_Count': rating_info['review_count'],
                'Specifications': json.dumps(specifications) if specifications else 'N/A',
                'Product_URL': product_url,
                'ASIN': asin,
                'Scraped_At': datetime.now().isoformat()
            }
            
            if asin:
                self.product_cache.put(asin, product_data)
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Error extracting product data: {str(e)}")
            return None
    
    async def scrape_search_results(self, search_input: str, max_pages: int = 2, 
                                  max_products: int = 50, extract_models: bool = True) -> List[Dict]:
        """Main scraping function with model number extraction"""
        start_time = time.time()
        self.products_data = []
        
        if not search_input.startswith('http'):
            search_url = f"https://www.amazon.in/s?k={search_input.replace(' ', '+')}"
        else:
            search_url = search_input
        
        self.logger.info(f"Starting enhanced scrape for: '{search_input}'")
        self.logger.info(f"Target: {max_products} products across {max_pages} pages")
        self.logger.info(f"Model number extraction: {'Enabled' if extract_models else 'Disabled'}")
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            products_found = 0
            
            for page in range(1, max_pages + 1):
                if products_found >= max_products:
                    break
                
                try:
                    page_url = f"{search_url}&page={page}" if page > 1 else search_url
                    if '?' not in search_url and page > 1:
                        page_url = f"{search_url}?page={page}"
                    
                    self.logger.info(f"Processing page {page}: {page_url}")
                    result = await crawler.arun(
                        url=page_url,
                        word_count_threshold=10,
                        bypass_cache=True,
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        wait_for="css:.s-result-item, [data-component-type='s-search-result']",
                        delay_before_return_html=random.uniform(2, 4)
                    )
                    
                    if not result.success:
                        self.logger.error(f"Failed to crawl page {page}: {result.error_message}")
                        continue
                    
                    soup = BeautifulSoup(result.html, 'html.parser')
                    containers = soup.select('div[data-component-type="s-search-result"], div.s-result-item[data-asin]')
                    
                    if not containers:
                        self.logger.warning(f"No products found on page {page}")
                        continue
                    
                    self.logger.info(f"Found {len(containers)} products on page {page}")
                    
                    for i, container in enumerate(containers):
                        if products_found >= max_products:
                            break
                        
                        try:
                            if extract_models:
                                product_data = await self.extract_product_info(container, crawler)
                            else:
                                product_data = await self.extract_product_info_basic(container)
                            
                            if not product_data:
                                continue
                            
                            if self.duplicate_tracker.is_duplicate(product_data):
                                self.metrics['duplicates_found'] += 1
                                continue
                            
                            self.products_data.append(product_data)
                            products_found += 1
                            
                            self.logger.info(f"Product {products_found}: {product_data['Product_Name'][:50]}...")
                            if product_data.get('Model_Number') and product_data['Model_Number'] != 'N/A':
                                self.logger.info(f"  Model: {product_data['Model_Number']}")
                            
                            await asyncio.sleep(0.1)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing product {i+1}: {str(e)}")
                            continue
                    
                    if page < max_pages:
                        delay = random.uniform(3, 6)
                        self.logger.info(f"Waiting {delay:.1f} seconds before next page...")
                        await asyncio.sleep(delay)
                
                except Exception as e:
                    self.logger.error(f"Error processing page {page}: {str(e)}")
                    continue
        
        self.metrics['processing_time'] = time.time() - start_time
        self.logger.info(f"Scraping completed in {self.metrics['processing_time']:.2f} seconds")
        return self.products_data
    
    async def extract_product_info_basic(self, container) -> Optional[Dict]:
        """Basic product extraction without model number lookup"""
        try:
            asin = container.get('data-asin', '')
            if asin:
                cached_product = self.product_cache.get(asin)
                if cached_product:
                    self.metrics['cache_hits'] += 1
                    return cached_product
            
            self.metrics['cache_misses'] += 1
            title_selectors = [
                'h2 a span', '.a-size-base-plus', '.a-size-medium', 'h2 a',
                '.s-size-mini span', 'a[data-csa-c-type="link"] span', '.a-text-normal', '.a-size-mini span'
            ]
            
            title = ""
            for selector in title_selectors:
                title_elem = container.select_one(selector)
                if title_elem:
                    title = self.clean_text(title_elem.get_text())
                    if title and len(title) > 10:
                        break
            
            if not title:
                return None
            
            link_selectors = ['h2 a', 'a[data-csa-c-type="link"]', 'a.a-link-normal']
            product_url = ""
            for selector in link_selectors:
                link_elem = container.select_one(selector)
                if link_elem and link_elem.get('href'):
                    product_url = urljoin(self.base_url, link_elem['href'])
                    break
            
            category = self.detect_category(title)
            brand = self.extract_brand_from_title(title)
            model_number = self.extract_model_number_from_title(title)
            specifications = self.extract_specifications(title, category)
            price_info = self.extract_price_info(container)
            rating_info = self.extract_rating_info(container)
            
            product_data = {
                'Product_Name': title,
                'Brand': brand,
                'Model_Number': model_number or 'N/A',
                'Category': category.title(),
                'Price': price_info['price'],
                'Original_Price': price_info['original_price'],
                'Discount': price_info['discount'],
                'Rating': rating_info['rating'],
                'Review_Count': rating_info['review_count'],
                'Specifications': json.dumps(specifications) if specifications else 'N/A',
                'Product_URL': product_url,
                'ASIN': asin,
                'Scraped_At': datetime.now().isoformat()
            }
            
            if asin:
                self.product_cache.put(asin, product_data)
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Error in basic extraction: {str(e)}")
            return None
    
    def get_price_statistics(self) -> Dict[str, float]:
        """Calculate price statistics"""
        prices = [self.extract_numeric_price(p.get('Price', '')) for p in self.products_data if self.extract_numeric_price(p.get('Price', '')) > 0]
        if not prices:
            return {'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        
        prices.sort()
        n = len(prices)
        return {
            'count': n,
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / n,
            'median': prices[n // 2] if n % 2 == 1 else (prices[n // 2 - 1] + prices[n // 2]) / 2
        }
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of products by category"""
        category_count = defaultdict(int)
        for product in self.products_data:
            category_count[product.get('Category', 'Unknown')] += 1
        return dict(category_count)
    
    def get_brand_distribution(self, top_n: int = 10) -> Dict[str, int]:
        """Get top N brands by product count"""
        brand_count = defaultdict(int)
        for product in self.products_data:
            brand_count[product.get('Brand', 'Unknown')] += 1
        return dict(heapq.nlargest(top_n, brand_count.items(), key=lambda x: x[1]))
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export scraped data to CSV"""
        if not self.products_data:
            self.logger.warning("No data to export to CSV")
            return ""
        
        if not filename:
            timestamp = int(time.time())
            filename = f"static/scraped_{timestamp}.csv"
        
        df = pd.DataFrame(self.products_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        self.logger.info(f"Data exported to {filename}")
        return filename
    
    def export_to_excel(self, filename: str = None) -> str:
        """Export to Excel with multiple sheets and analytics"""
        if not self.products_data:
            self.logger.warning("No data to export to Excel")
            return ""
        
        if not filename:
            timestamp = int(time.time())
            filename = f"static/scraped_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df = pd.DataFrame(self.products_data)
            df['Price_Numeric'] = df['Price'].apply(self.extract_numeric_price)
            df['Has_Model_Number'] = df['Model_Number'].apply(lambda x: 'Yes' if x != 'N/A' else 'No')
            df.to_excel(writer, sheet_name='Products', index=False)
            
            cat_dist = self.get_category_distribution()
            cat_df = pd.DataFrame(list(cat_dist.items()), columns=['Category', 'Count'])
            cat_df.to_excel(writer, sheet_name='Category_Analysis', index=False)
            
            brand_dist = self.get_brand_distribution()
            brand_df = pd.DataFrame(list(brand_dist.items()), columns=['Brand', 'Count'])
            brand_df.to_excel(writer, sheet_name='Brand_Analysis', index=False)
            
            price_stats = self.get_price_statistics()
            stats_df = pd.DataFrame([price_stats])
            stats_df.to_excel(writer, sheet_name='Price_Statistics', index=False)
        
        self.logger.info(f"Data exported to {filename}")
        return filename

# Global scraper instance
scraper = AdvancedAmazonScraper()

@app.route('/')
def index():
    """Serve the main page from the current folder"""
    return send_file("index.html")

@app.route('/api/scrape', methods=['POST'])
async def scrape():
    """API endpoint to start scraping"""
    try:
        data = request.get_json()
        search_input = data.get('search_query', '')
        max_pages = int(data.get('max_pages', 2))
        max_products = int(data.get('max_products', 50))
        extract_models = data.get('extract_models', True)
        
        if not search_input:
            return jsonify({'error': 'Search query is required'}), 400
        
        products = await scraper.scrape_search_results(search_input, max_pages, max_products, extract_models)
        
        price_stats = scraper.get_price_statistics()
        category_dist = scraper.get_category_distribution()
        brand_dist = scraper.get_brand_distribution()
        
        response = {
            'products': products,
            'metrics': scraper.metrics,
            'price_statistics': price_stats,
            'category_distribution': category_dist,
            'brand_distribution': brand_dist
        }
        
        return jsonify(response)
    
    except Exception as e:
        scraper.logger.error(f"Error in scrape API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export():
    """Export scraped data to CSV or Excel"""
    try:
        data = request.get_json()
        export_type = data.get('export_type', 'csv')
        filename = scraper.export_to_csv() if export_type == 'csv' else scraper.export_to_excel()
        if not filename:
            return jsonify({'error': 'No data to export'}), 400
        return jsonify({'filename': filename})
    except Exception as e:
        scraper.logger.error(f"Error in export API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (CSV/Excel downloads)"""
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
