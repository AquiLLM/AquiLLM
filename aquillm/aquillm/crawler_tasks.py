import structlog
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from celery import shared_task
from celery.states import state, STARTED, SUCCESS, FAILURE
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import DatabaseError
from django.contrib.auth import get_user_model

# Channels imports for WebSocket communication
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# Trafilatura imports
from trafilatura import fetch_url, extract, extract_metadata

# Local imports
from .models import RawTextDocument, Collection, DuplicateDocumentError
from .celery import app # Ensure Celery app is imported

logger = structlog.stdlib.get_logger(__name__)

# Constants
MIN_TEXT_LENGTH = 50 # Minimum characters to consider extraction successful
SELENIUM_WAIT_TIME = 10 # Seconds to wait for dynamic content in Selenium


def _strip_query(url: str) -> str:
    """Return URL without query string for logging."""
    parsed = urlparse(url)
    return parsed._replace(query="", fragment="").geturl()


def is_same_domain(base_url, link_url):
    """Checks if a link URL belongs to the same domain as the base URL."""
    base_domain = urlparse(base_url).netloc
    link_domain = urlparse(link_url).netloc
    return link_domain == base_domain

def find_links(html_content, base_url):
    """Finds valid, same-domain absolute links within HTML content."""
    links = set()
    soup = BeautifulSoup(html_content, 'html.parser')
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        # Resolve relative URLs and ensure they are HTTP/HTTPS
        absolute_url = urljoin(base_url, href)
        parsed_url = urlparse(absolute_url)
        if parsed_url.scheme in ['http', 'https'] and is_same_domain(base_url, absolute_url):
            # Remove fragments
            links.add(parsed_url._replace(fragment="").geturl())
    return links

# Helper function to send status updates via WebSocket
def send_crawl_status(user_id: int, task_id: str, message_type: str, payload: dict):
    """Sends a status update message to the user-specific WebSocket group."""
    try:
        channel_layer = get_channel_layer()
        group_name = f'crawl-status-{user_id}'
        logger.debug("obs.crawl.ws_send", task_id=task_id, user_id=user_id, message_type=message_type, group_name=group_name)
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                'type': 'crawl.task.update', # Corresponds to the method in the Consumer
                'data': {
                    'task_id': task_id,
                    'message_type': message_type, # e.g., 'progress', 'success', 'error'
                    **payload, # Include specific data like progress percentage, error message, etc.
                }
            }
        )
    except Exception as e:
        # Log if sending the WebSocket message fails, but don't let it crash the task
        logger.error("obs.crawl.ws_send_error", task_id=task_id, user_id=user_id, error_type=type(e).__name__, error=str(e), exc_info=False)


@app.task(bind=True, track_started=True, serializer='json')
def crawl_and_ingest_webpage(self, initial_url: str, collection_id: int, user_id: int, max_depth: int = 1):
    """
    Celery task to crawl a webpage, follow links (1 level deep, same domain),
    extract text using Trafilatura (with Selenium fallback), and save as a RawTextDocument.
    """
    task_id = str(self.request.id) # Ensure task_id is a string for consistency
    logger.info("obs.crawl.start", task_id=task_id, url=_strip_query(initial_url), collection_id=collection_id, user_id=user_id, max_depth=max_depth)
    self.update_state(state=STARTED, meta={'current_url': initial_url, 'progress': 0, 'task_id': task_id})
    # Send initial start message via WebSocket
    send_crawl_status(user_id, task_id, 'crawl.start', {'initial_url': initial_url, 'message': 'Crawl initiated...'})

    try:
        collection = Collection.objects.get(pk=collection_id)
        user = get_user_model().objects.get(pk=user_id)
    except ObjectDoesNotExist as e:
        error_msg = 'Collection or User not found.'
        logger.error("obs.crawl.error", task_id=task_id, error=error_msg, error_type=type(e).__name__)
        self.update_state(state=FAILURE, meta={'error': error_msg, 'task_id': task_id})
        send_crawl_status(user_id, task_id, 'crawl.error', {'error': error_msg})
        return {'error': error_msg}

    visited_urls = set()
    urls_to_visit = [(initial_url, 0)] # (url, depth)
    all_text_content = []
    page_title = initial_url # Fallback title
    processed_count = 0
    total_urls_found = 1 # Start with the initial URL

    selenium_driver = None # Initialize driver variable

    try:
        while urls_to_visit:
            current_url, current_depth = urls_to_visit.pop(0)
            processed_count += 1
            progress = int((processed_count / max(total_urls_found, 1)) * 90) # Avoid division by zero; 90% for crawling
            status_message = f"Processing URL {processed_count}/{total_urls_found}: {current_url}"
            self.update_state(state=STARTED, meta={'current_url': current_url, 'progress': progress, 'message': status_message, 'task_id': task_id})
            send_crawl_status(user_id, task_id, 'crawl.progress', {'progress': progress, 'message': status_message})
            logger.info("obs.crawl.status", task_id=task_id, url=_strip_query(current_url), processed=processed_count, total=total_urls_found, progress=progress)

            if current_url in visited_urls or current_depth > max_depth:
                reason = "visited" if current_url in visited_urls else "max_depth_exceeded"
                logger.debug("obs.crawl.skip", task_id=task_id, url=_strip_query(current_url), reason=reason)
                continue

            visited_urls.add(current_url)
            extracted_text = None
            html_content_for_links = None
            new_links_found = set()

            # --- Attempt 1: Trafilatura ---
            try:
                logger.debug("obs.crawl.fetch", task_id=task_id, url=_strip_query(current_url), method="trafilatura")
                downloaded = fetch_url(current_url)
                if downloaded:
                    logger.debug("obs.crawl.fetch", task_id=task_id, url=_strip_query(current_url), method="trafilatura", status="success")
                    text = extract(downloaded, include_comments=False, include_tables=True)
                    if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                        extracted_text = text.strip()
                        html_content_for_links = downloaded # Use downloaded content to find links
                        logger.info("obs.crawl.extract", task_id=task_id, url=_strip_query(current_url), method="trafilatura", char_count=len(extracted_text))
                        if current_depth == 0:
                            metadata = extract_metadata(downloaded)
                            if metadata and metadata.title:
                                page_title = metadata.title.strip().replace('\n', ' ').replace('\r', '')
                                logger.info("obs.crawl.extract", task_id=task_id, url=_strip_query(current_url), title=page_title)
                    else:
                        logger.warning("obs.crawl.extract_warning", task_id=task_id, url=_strip_query(current_url), method="trafilatura", char_count=len(text.strip()) if text else 0, reason="insufficient_text")
                else:
                    logger.warning("obs.crawl.extract_warning", task_id=task_id, url=_strip_query(current_url), method="trafilatura", reason="fetch_returned_none")

            except Exception as e:
                logger.warning("obs.crawl.fetch_error", task_id=task_id, url=_strip_query(current_url), method="trafilatura", error_type=type(e).__name__, error=str(e), exc_info=False)

            # --- Attempt 2: Selenium (if Trafilatura failed or insufficient) ---
            if not extracted_text:
                logger.info("obs.crawl.fetch", task_id=task_id, url=_strip_query(current_url), method="selenium", reason="trafilatura_fallback")
                try:
                    if selenium_driver is None: # Initialize driver only if needed
                        chrome_options = ChromeOptions()
                        chrome_options.add_argument("--headless") # Run headless
                        chrome_options.add_argument("--no-sandbox")
                        chrome_options.add_argument("--disable-dev-shm-usage")
                        # Use webdriver-manager to automatically handle driver download/updates
                        service = ChromeService(ChromeDriverManager().install())
                        selenium_driver = webdriver.Chrome(service=service, options=chrome_options)
                        selenium_driver.implicitly_wait(SELENIUM_WAIT_TIME) # Implicit wait
                        logger.info("obs.crawl.status", task_id=task_id, status="selenium_initialized")

                    selenium_driver.get(current_url)
                    # Consider adding explicit waits here if needed for specific dynamic content
                    # WebDriverWait(selenium_driver, SELENIUM_WAIT_TIME).until(...)

                    page_source = selenium_driver.page_source
                    if page_source:
                        # Try extracting text from Selenium source using Trafilatura again
                        text = extract(page_source, include_comments=False, include_tables=True)
                        if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                            extracted_text = text.strip()
                            html_content_for_links = page_source # Use page source to find links
                            logger.info("obs.crawl.extract", task_id=task_id, url=_strip_query(current_url), method="selenium_trafilatura", char_count=len(extracted_text))
                            if current_depth == 0:
                                sel_title = selenium_driver.title
                                if sel_title:
                                    page_title = sel_title.strip().replace('\n', ' ').replace('\r', '')
                                    logger.info("obs.crawl.extract", task_id=task_id, url=_strip_query(current_url), method="selenium", title=page_title)
                        else:
                             # Fallback: Get text directly from body (less reliable)
                             try:
                                 body_text = selenium_driver.find_element("tag name", "body").text
                                 if body_text and len(body_text.strip()) >= MIN_TEXT_LENGTH:
                                      extracted_text = body_text.strip()
                                      html_content_for_links = page_source
                                      logger.info("obs.crawl.extract", task_id=task_id, url=_strip_query(current_url), method="selenium_body", char_count=len(extracted_text))
                                      if current_depth == 0 and not page_title: # Update title if not already set
                                          sel_title = selenium_driver.title
                                          if sel_title:
                                              page_title = sel_title.strip().replace('\n', ' ').replace('\r', '')
                                              logger.info("obs.crawl.extract", task_id=task_id, url=_strip_query(current_url), method="selenium", title=page_title)
                                 else:
                                     logger.warning("obs.crawl.extract_warning", task_id=task_id, url=_strip_query(current_url), method="selenium_body", char_count=len(body_text.strip()) if body_text else 0, reason="insufficient_text")
                             except Exception as body_e:
                                 logger.warning("obs.crawl.fetch_error", task_id=task_id, url=_strip_query(current_url), method="selenium_body", error_type=type(body_e).__name__, error=str(body_e), exc_info=False)

                    else:
                        logger.warning("obs.crawl.extract_warning", task_id=task_id, url=_strip_query(current_url), method="selenium", reason="empty_page_source")

                except WebDriverException as e:
                    logger.error("obs.crawl.fetch_error", task_id=task_id, url=_strip_query(current_url), method="selenium", error_type=type(e).__name__, error=str(e), exc_info=False)
                except Exception as e:
                    logger.error("obs.crawl.fetch_error", task_id=task_id, url=_strip_query(current_url), method="selenium", error_type=type(e).__name__, error=str(e), exc_info=False)


            # --- Process extracted text and find links ---
            if extracted_text:
                all_text_content.append(f"\n\n--- Source: {current_url} ---\n\n{extracted_text}")
                if current_depth < max_depth and html_content_for_links:
                    try:
                        found_links = find_links(html_content_for_links, current_url)
                        # Add only new, unvisited links that are not already in the queue
                        new_links_found = {link for link in found_links if link not in visited_urls and not any(item[0] == link for item in urls_to_visit)}
                        if new_links_found:
                            logger.info("obs.crawl.links_found", task_id=task_id, url=_strip_query(current_url), depth=current_depth, link_count=len(new_links_found))
                            for link in new_links_found:
                                urls_to_visit.append((link, current_depth + 1))
                            total_urls_found = processed_count + len(urls_to_visit) # Update total for progress calc
                    except Exception as link_e:
                        logger.error("obs.crawl.error", task_id=task_id, url=_strip_query(current_url), error_type=type(link_e).__name__, error=str(link_e), exc_info=False)
            else:
                 logger.error("obs.crawl.fetch_error", task_id=task_id, url=_strip_query(current_url), reason="both_methods_failed")


        # --- Combine and Save ---
        if all_text_content:
            final_text = "".join(all_text_content).strip()
            logger.info("obs.crawl.complete", task_id=task_id, url=_strip_query(initial_url), char_count=len(final_text))
            self.update_state(state=STARTED, meta={'current_url': 'Saving Document', 'progress': 95, 'message': f"Crawling complete. Total text length: {len(final_text)}. Saving document...", 'task_id': task_id})
            send_crawl_status(user_id, task_id, 'crawl.progress', {'progress': 95, 'message': f"Crawling complete. Total text length: {len(final_text)}. Saving document..."})
            try:
                doc = RawTextDocument(
                    title=page_title,
                    full_text=final_text,
                    collection=collection,
                    ingested_by=user,
                    source_url=initial_url # Link back to the initial URL requested
                )
                doc.save() # This triggers the chunking task via the model's save method
                doc_id_str = str(doc.id)
                doc_title = doc.title
                logger.info("obs.crawl.doc_saved", task_id=task_id, doc_id=doc_id_str, url=_strip_query(initial_url))
                self.update_state(state=SUCCESS, meta={'document_id': doc_id_str, 'title': doc_title, 'progress': 100, 'task_id': task_id})
                send_crawl_status(user_id, task_id, 'crawl.success', {'document_id': doc_id_str, 'title': doc_title, 'message': 'Crawl and save successful.'})
                return {'document_id': doc_id_str, 'title': doc_title}

            except DuplicateDocumentError as e:
                error_msg = e.message
                logger.warning("obs.crawl.doc_duplicate", task_id=task_id, url=_strip_query(initial_url), error=error_msg)
                self.update_state(state=FAILURE, meta={'error': error_msg, 'task_id': task_id})
                send_crawl_status(user_id, task_id, 'crawl.error', {'error': error_msg})
                return {'error': error_msg}
            except ValidationError as e:
                 error_msg = "; ".join([f'{k}: {v[0]}' for k, v in e.message_dict.items()]) if hasattr(e, 'message_dict') else ". ".join(e.messages)
                 error_msg = f'Validation Error: {error_msg}'
                 logger.error("obs.crawl.error", task_id=task_id, url=_strip_query(initial_url), error_type=type(e).__name__, error=error_msg, exc_info=True)
                 self.update_state(state=FAILURE, meta={'error': error_msg, 'task_id': task_id})
                 send_crawl_status(user_id, task_id, 'crawl.error', {'error': error_msg})
                 return {'error': error_msg}
            except DatabaseError as e:
                error_msg = 'Database error during save.'
                logger.error("obs.crawl.error", task_id=task_id, url=_strip_query(initial_url), error_type=type(e).__name__, error=str(e), exc_info=True)
                self.update_state(state=FAILURE, meta={'error': error_msg, 'task_id': task_id})
                send_crawl_status(user_id, task_id, 'crawl.error', {'error': error_msg})
                return {'error': error_msg}
            except Exception as e:
                error_msg = 'Unexpected error during save.'
                logger.error("obs.crawl.error", task_id=task_id, url=_strip_query(initial_url), error_type=type(e).__name__, error=str(e), exc_info=True)
                self.update_state(state=FAILURE, meta={'error': error_msg, 'task_id': task_id})
                send_crawl_status(user_id, task_id, 'crawl.error', {'error': error_msg})
                return {'error': error_msg}
        else:
            error_msg = 'No text content could be extracted.'
            logger.error("obs.crawl.error", task_id=task_id, url=_strip_query(initial_url), error=error_msg)
            self.update_state(state=FAILURE, meta={'error': error_msg, 'task_id': task_id})
            send_crawl_status(user_id, task_id, 'crawl.error', {'error': error_msg})
            return {'error': error_msg}

    except Exception as e:
        logger.error("obs.crawl.error", task_id=task_id, error_type=type(e).__name__, error=str(e), exc_info=True)
        self.update_state(state=FAILURE, meta={'error': str(e), 'task_id': task_id})
        send_crawl_status(user_id, task_id, 'crawl.error', {'error': str(e)})
        return {'error': str(e)}
    finally:
        # Ensure Selenium driver is closed if it was opened
        if selenium_driver:
            try:
                selenium_driver.quit()
                logger.info("obs.crawl.status", task_id=task_id, status="selenium_closed")
            except Exception as e:
                logger.error("obs.crawl.error", task_id=task_id, error_type=type(e).__name__, error=str(e), exc_info=False)
