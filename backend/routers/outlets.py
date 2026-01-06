import os
import json
import httpx
import google.generativeai as genai
from bs4 import BeautifulSoup
import html # For escaping content in f-strings
import asyncio # Added for digest parallel requests
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import select, distinct
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from database import AsyncSessionLocal
from database import AsyncSessionLocal
from models import NewsOutlet, User, Country, CityMetadata
from dependencies import get_current_user
import scraper_engine # New Import

from datetime import datetime, timedelta
import re

router = APIRouter()

# --- Pydantic Schemas ---
class OutletCreate(BaseModel):
    name: str
    country_code: str
    city: str
    lat: Optional[float] = 0.0
    lng: Optional[float] = 0.0
    url: Optional[str] = None
    type: Optional[str] = "Unknown" # Print, Online, TV, Radio
    popularity: Optional[int] = 5
    focus: Optional[str] = "Local"

class OutletRead(BaseModel):
    id: int
    name: str
    country_code: str
    city: str
    lat: float
    lng: float
    url: Optional[str] = None
    type: Optional[str] = "Unknown"
    url: Optional[str] = None
    type: Optional[str] = "Unknown"
    origin: Optional[str] = "auto"
    popularity: Optional[int] = 5
    focus: Optional[str] = "Local"
    
    class Config:
        from_attributes = True

class GeocodeRequest(BaseModel):
    city: str
    country: str

class GeocodeResponse(BaseModel):
    lat: float
    lng: float

from google.api_core.exceptions import ResourceExhausted

class CityDiscoveryRequest(BaseModel):
    city: str
    country: str
    lat: Optional[float] = 0.0
    lng: Optional[float] = 0.0
    force_refresh: bool = False

class ImportUrlRequest(BaseModel):
    url: str
    city: str
    country: str
    lat: Optional[float] = 0.0

class CityInfoResponse(BaseModel):
    population: str
    description: str
    ruling_party: str
    flag_url: Optional[str] = None
    lng: Optional[float] = 0.0
    lung: Optional[float] = 0.0
    instructions: Optional[str] = None

# --- Constants ---

POLITICS_OPERATIONAL_DEFINITION = """
Politics label spec (operational)
Label name: POLITICS

Core criterion:
Assign POLITICS if the primary focus of the article is power, governance, or collective decision-making carried out by political institutions/actors, or the processes that select/control them.
“Primary focus” = the main story would still be the same if you removed all non-political details; politics is not just a cameo.

Include if ANY of these is the main subject:
A) Government & institutions (domestic)
- Executive actions: cabinet decisions, ministries, agencies, regulators acting in official capacity
- Legislature: bills, votes, committees, parliamentary negotiations
- Public administration: government programs, budgets, procurement policy (not company-specific business news)
- Local government: mayors, councils, regional authorities, public service governance

B) Elections & party politics
- Elections, campaigns, polling, debates, candidate selection, coalition talks
- Party leadership, internal party conflicts when politically consequential
- Political strategy, messaging, endorsements

C) Public policy (substance + debate)
- Policy proposals, reforms, regulation, taxation, welfare, healthcare policy, education policy, climate policy, etc.
- Political conflict over policy (who supports/opposes; parliamentary dynamics; veto threats)

D) Political accountability & legitimacy
- Resignations, impeachments, no-confidence votes
- Ethics, corruption, conflicts of interest when tied to governance (not just criminal detail)
- Constitutional crises, institutional clashes, rule-of-law disputes

E) International politics & diplomacy
- Treaties, summits, sanctions, foreign policy statements
- Diplomatic incidents, recognition disputes, geopolitical negotiations

F) Civil liberties & rights as political contestation
- Protests, civil society actions, strikes when framed around policy/government power
- Major court rulings when they reshape governance or political rights (elections, constitutional issues)

Exclude (unless politics is clearly primary):
1) Crime / courts: If it’s mainly “who did what, evidence, trial details,” label CRIME/LAW, not POLITICS. Exception: if the case directly affects governance.
2) Business / economy: Market moves, company earnings, mergers → BUSINESS. Exception: sanctions, regulation, antitrust, budgets where the policy process is central → POLITICS.
3) Disasters / weather / accidents: If the focus is the event itself → DISASTER. Exception: political accountability/policy response dominates.
4) Culture / celebrity: Politicians as celebrities (personal life) → ENTERTAINMENT unless tied to office, campaign, or legitimacy.
5) Sports: Sports story with a politician quote is still SPORTS unless it becomes policy.

Decision rules:
Rule P1 — Actor × Action (strong signal): If article contains political actors/institutions AND governance actions, assign POLITICS.
Rule P2 — Elections/party process (strong signal): If the main content concerns elections, campaigns, polling, coalitions, party leadership, assign POLITICS.
Rule P3 — Policy conflict frame (medium signal): If the article is structured as policy debate, assign POLITICS.
Rule P4 — International statecraft (strong signal): If it involves states/IGOs and diplomatic/military/economic coercion instruments, assign POLITICS.
Rule P5 — “Mention-only” veto: If political entity is mentioned but not central, do NOT assign POLITICS.
"""

class PoliticsAssessmentRequest(BaseModel):
    url: str
    title: str
    content: Optional[str] = None # Optional, if frontend already has it or we re-fetch

class PoliticsAssessmentResponse(BaseModel):
    is_politics: bool
    confidence: int # 0-100
    reasoning: str
    labels: List[str] # e.g. ["POLITICS", "HEALTH"]

# --- Helpers (Moved to scraper_engine) ---
# parse_romanian_date and extract_date_from_url removed from here


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def gemini_discover_city_outlets(city: str, country: str, lat: float, lng: float, api_key: str) -> List[OutletCreate]:
    if not api_key: return []
    genai.configure(api_key=api_key)

    prompt = f"""
    You are a news outlet discovery expert. Try finding the townhall page of the {city}, {country} and see if there is information regarding the local media in {city}. For example, in "Sibiu", there is this page "https://www.sibiu.ro/sibiu/media"  which lists all media produced in the city.
    If you cannot find such a page, then find the top 15-20 most relevant local news outlets (Newspapers, TV Stations, Radio, Online Portals) based in or covering: {city}, {country}.
    Do not ignore national outlets if they are based in the same location: {city}, {country}
    Focus on finding and validating live Website URLs. Do not return outdated, non-loading or broken links. 
    After finding the outlets, assign a popularity score from 1 to 10 based on the outlet's reputation and reach.
    
    Return a strictly valid JSON list. Example:
    [
        {{ "name": "Monitorul de Cluj", "url": "https://www.monitorulcj.ro", "type": "Online", "popularity": 10, "focus": "Local" }},
        {{ "name": "Radio Cluj", "url": "http://radiocluj.ro", "type": "Radio", "popularity": 7, "focus": "Local and National" }}
    ]
    """
    
    print(f"DEBUG: Starting Gemini Discovery for {city}, {country}")
    model = genai.GenerativeModel('gemini-flash-latest')
    try:
        response = await model.generate_content_async(prompt, generation_config={"max_output_tokens": 4000})
        text = response.text.strip()
        print(f"DEBUG: Gemini response received. Length: {len(text)}")
        
        # Robust JSON finding using Regex
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            text = match.group(0)
            print("DEBUG: JSON block found.")
        else:
            print(f"DEBUG: No JSON block found in response: {text[:100]}...")
            return []
        
        try:
            data = json.loads(text)
            print(f"DEBUG: JSON parsed successfully. {len(data)} items.")
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON Parsed Error: {e}")
            # Try to fix common trailing comma issues or markdown
            return []
        
        def safe_int(val):
            try: return int(val)
            except: return 5

        outlets = [OutletCreate(
            name=d['name'], 
            city=city, 
            country_code="RO" if "Romania" in country else "XX",
            url=d.get('url'),
            type=d.get('type', 'Online'),
            popularity=safe_int(d.get('popularity', 5)),
            focus=d.get('focus', 'Local'),
            lat=lat,
            lng=lng
        ) for d in data]
        print(f"DEBUG: Processed {len(outlets)} outlets.")
        return outlets
    except Exception as e:
        print(f"DEBUG: Gemini Discovery Critical Error for {city}: {e}")
        import traceback
        traceback.print_exc()
        return []
        # Log raw text for debugging if available
        try: print(f"Raw Response: {response.text}") 
        except: pass
        return []

async def gemini_scrape_outlets(html_content: str, city: str, country: str, lat: float, lng: float, api_key: str, instructions: str = None) -> List[OutletCreate]:
    if not api_key: return []
    genai.configure(api_key=api_key)

    # Truncate HTML to avoid token limits (approx 30k chars is usually enough for structure)
    html_sample = html_content[:50000]

    prompt = f"""
    Analyze this HTML content from a website ({instructions or "General Analysis"}).
    
    Goal: Identify the News Outlet(s) associated with this page.
    
    Scenario A (Single Outlet): The website ITSELF is a news outlet (newspaper, blog, TV station).
    -> Return it as a single entry.
    
    Scenario B (Directory): The website contains a LIST of OTHER news outlets.
    -> Return the list of extracted outlets.
    
    Context: We are looking for media related to {city}, {country}.
    
    Return a JSON list:
    [
        {{ "name": "Outlet Name", "url": "https://full.url.com", "type": "Online" }}
    ]
    """
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = await model.generate_content_async([prompt, html_sample])
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return [OutletCreate(
            name=d['name'], 
            city=city, 
            country_code="RO" if "Romania" in country else "XX",
            url=d.get('url'),
            type=d.get('type', 'Online'),
            lat=lat,
            lng=lng
        ) for d in data]
    except ResourceExhausted:
         raise # Re-raise 429
    except Exception as e:
        print(f"Gemini Scrape Error: {e}")
        return []

@router.post("/outlets/discover_city", response_model=List[OutletRead])
async def discover_city_outlets(req: CityDiscoveryRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Finds outlets for a specific city.
    Checks DB first. If not found OR force_refresh is True, uses AI to discover.
    """
    existing = []
    print(f"DEBUG: Request for {req.city}, force={req.force_refresh}")
    if not req.force_refresh:
        result = await db.execute(select(NewsOutlet).where(NewsOutlet.city.ilike(req.city)))
        existing = result.scalars().all()
        print(f"DEBUG: Found {len(existing)} existing outlets in DB.")
        if existing:
            return existing
    
    # Auto-Discover
    print(f"Discovering outlets for city: {req.city}, {req.country} (Force: {req.force_refresh})")
    
    try:
        discovered = await gemini_discover_city_outlets(req.city, req.country, req.lat, req.lng, api_key=current_user.gemini_api_key)
    except ResourceExhausted:
         raise HTTPException(status_code=429, detail="AI Quota Exceeded. Please try again later.")
    except Exception as e:
        print(f"Discovery Error: {e}")
        # If we have existing data and discovery failed, return existing
        if existing: return existing
        return []

    saved_outlets = []
    
    # If refreshing, we might get duplicates. Only add new ones.
    # We re-fetch existing to be sure
    result = await db.execute(select(NewsOutlet).where(NewsOutlet.city.ilike(req.city)))
    current_db_outlets = result.scalars().all()
    current_urls = {o.url for o in current_db_outlets if o.url}
    current_names = {o.name.lower() for o in current_db_outlets}

    for disc in discovered:
        # Check duplicate by URL or Name
        if disc.url and disc.url in current_urls: continue
        if disc.name.lower() in current_names: continue
        
        db_outlet = NewsOutlet(
            name=disc.name,
            country_code="RO" if "Romania" in req.country or "RO" in req.country else "XX", 
            city=disc.city,
            lat=disc.lat,
            lng=disc.lng,
            url=disc.url,
            type=disc.type,
            popularity=disc.popularity,
            focus=disc.focus 
        )
        db.add(db_outlet)
        saved_outlets.append(db_outlet)
    
    if saved_outlets:
        await db.commit()
    
    # Return updated list
    result = await db.execute(select(NewsOutlet).where(NewsOutlet.city.ilike(req.city)))
    final_outlets = result.scalars().all()
    print(f"DEBUG: Returning {len(final_outlets)} outlets to frontend for {req.city}.")
    return final_outlets

class OutletUpdate(BaseModel):
    url: Optional[str] = None
    name: Optional[str] = None

@router.delete("/outlets/{outlet_id}")
async def delete_outlet(outlet_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = await db.execute(select(NewsOutlet).where(NewsOutlet.id == outlet_id))
    outlet = result.scalars().first()
    if not outlet:
        raise HTTPException(status_code=404, detail="Outlet not found")
    
    await db.delete(outlet)
    await db.commit()
    return {"status": "deleted", "id": outlet_id}

@router.put("/outlets/{outlet_id}", response_model=OutletRead)
async def update_outlet(outlet_id: int, update_data: OutletUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = await db.execute(select(NewsOutlet).where(NewsOutlet.id == outlet_id))
    outlet = result.scalars().first()
    if not outlet:
        raise HTTPException(status_code=404, detail="Outlet not found")
    
    if update_data.url is not None:
        outlet.url = update_data.url
    if update_data.name is not None:
        outlet.name = update_data.name
        
    await db.commit()
    await db.refresh(outlet)
    return outlet

@router.post("/outlets/import_from_url", response_model=List[OutletRead])
async def import_outlets_from_url(req: ImportUrlRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Scrapes a provided URL and uses AI to extract outlet information, saving it to the DB.
    """
    print(f"Importing for {req.city} from {req.url}")
    
    # 1. Fetch URL
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=15, headers=headers) as client:
            response = await client.get(req.url)
            response.raise_for_status()
            html_content = response.text
    except Exception as e:
        print(f"Fetch failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    # 2. Extract
    extracted = await gemini_scrape_outlets(html_content, req.city, req.country, req.lat, req.lng, api_key=current_user.gemini_api_key, instructions=req.instructions)
    
    # 3. Save
    saved_outlets = []
    for out in extracted:
        # Check if exists to avoid dupes? (Simple check: name + city)
        result = await db.execute(select(NewsOutlet).where(NewsOutlet.name == out.name, NewsOutlet.city == req.city))
        if result.scalars().first():
            continue # Skip duplicate
            
        db_outlet = NewsOutlet(
            name=out.name,
            country_code="RO" if "Romania" in req.country or "RO" in req.country else "XX",
            city=out.city,
            lat=out.lat,
            lng=out.lng,
            url=out.url,
            type=out.type or "Unknown", # Ensure type
            origin="manual",
            popularity=out.popularity,
            focus=out.focus
        )
        db.add(db_outlet)
        saved_outlets.append(db_outlet)
        
    if saved_outlets:
        await db.commit()
    
    # Return all outlets for city (including old ones)
    result = await db.execute(select(NewsOutlet).where(NewsOutlet.city.ilike(req.city)))
    return result.scalars().all()

@router.get("/outlets/cities/list", response_model=List[str])
async def list_cities_with_outlets(db: Session = Depends(get_db)):
    """Returns a list of distinct city names that have stored outlets."""
    result = await db.execute(select(distinct(NewsOutlet.city)))
    cities = result.scalars().all()
    return [c for c in cities if c]



# --- News Digest Agent ---
from bs4 import BeautifulSoup
from models import NewsDigest

class KeywordData(BaseModel):
    word: str
    importance: int # 1-100
    type: str 
    sentiment: str
    source_urls: Optional[List[str]] = [] # New: Links to specific source articles

class ArticleMetadata(BaseModel):
    title: str
    url: str
    source: str
    image_url: Optional[str] = None
    date_str: Optional[str] = None
    relevance_score: Optional[int] = 0
    scores: Optional[Dict[str, Any]] = {}
    ai_verdict: Optional[str] = None # New field for AI Title Check status
    
class DigestResponse(BaseModel):
    digest: str
    articles: List[ArticleMetadata]
    analysis_source: Optional[List[KeywordData]] = []
    analysis_digest: Optional[List[KeywordData]] = []

@router.post("/outlets/assess_article", response_model=PoliticsAssessmentResponse)
async def assess_article_politics(req: PoliticsAssessmentRequest, current_user: User = Depends(get_current_user)):
    """
    Evaluates an article against the operational definition of POLITICS using Gemini.
    """
    if not current_user.gemini_api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key required")
    
    genai.configure(api_key=current_user.gemini_api_key)
    
    # Needs content. If not provided, fetch it (snippet).
    article_text = req.content
    if not article_text or len(article_text) < 100:
        # Fetch ephemeral
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            try:
                resp = await client.get(req.url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    # Basic extraction
                    article_text = soup.get_text(separator=' ', strip=True)[:15000] # Limit context
            except:
                article_text = "Content unavailable. Rely on Title."

    prompt = f"""
    You are an expert political analyst system.
    Evaluate the following article against the provided OPERATIONAL DEFINITION OF POLITICS.
    
    DEFINITION:
    {POLITICS_OPERATIONAL_DEFINITION}
    
    ARTICLE TITLE: {req.title}
    ARTICLE CONTENT (Snippet):
    {article_text[:10000]}
    
    TASK:
    1. Determine if this article qualifies as POLITICS based on the inclusion/exclusion criteria.
    2. Provide a Confidence score (0-100).
    3. Choose appropriate labels (e.g., POLITICS, BUSINESS, CRIME).
    4. Provide brief reasoning (max 1 sentence).
    
    Return JSON:
    {{
        "is_politics": true/false,
        "confidence": 90,
        "labels": ["POLITICS", "ECONOMY"],
        "reasoning": "Primary focus is on government budget approval (Rule P1)."
    }}
    """
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = await model.generate_content_async(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return PoliticsAssessmentResponse(**data)
    except Exception as e:
        print(f"Assessment Error: {e}")
        return PoliticsAssessmentResponse(is_politics=False, confidence=0, reasoning=f"Error: {str(e)}", labels=[])

# Helper to log to stream from async function
# We will return a tuple (result_map, error_msg)


async def batch_verify_titles_debug(titles_map: Dict[int, str], definition: str, api_key: str) -> tuple[Dict[str, bool], str]:
    if not api_key:
        print("DEBUG: missing API key for batch_verify_titles_debug")
        return {}, "Missing API Key"
    
    print(f"DEBUG: Using API Key: {api_key[:4]}...{api_key[-4:]}")
    genai.configure(api_key=api_key)
    
    items_str = "\n".join([f"{idx}. {title}" for idx, title in titles_map.items()])
    
    prompt = f"""
    You are an expert political analyst.
    Classify the following article titles as "POLITICS" (True) or "NOT POLITICS" (False) based on the provided Operational Definition.
    
    DEFINITION:
    {definition}
    
    TITLES:
    {items_str}
    
    TASK:
    Return a raw JSON object mapping the EXACT PROVIDED ID (e.g. {list(titles_map.keys())[0]}) to a boolean verdict.
    Do NOT re-index the items. Use the numbers provided in the input list.
    Example: {{"{list(titles_map.keys())[0]}": true, ...}}
    
    STRICTLY RETURN JSON ONLY.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = await model.generate_content_async(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        
        with open("debug_ai.log", "a") as f:
             f.write(f"\n--- BATCH ---\nResponse: {text[:200]}...\n")

        # Robust Parsing
        try:
            result_map = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: Try regex to find JSON-like structure
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result_map = json.loads(match.group(0))
            else:
                raise ValueError("Could not extract JSON")

        # Convert all keys to strings for consistent comparison
        result_map = {str(k): v for k, v in result_map.items()}
        input_keys = [str(k) for k in titles_map.keys()]
        
        # Mismatch Check & Fallback
        if not any(k in result_map for k in input_keys) and len(result_map) == len(input_keys):
            with open("debug_ai.log", "a") as f:
                f.write("Mismatch detected. Applying fallback.\n")
            # Assume order is preserved
            result_values = list(result_map.values())
            result_map = {input_keys[i]: result_values[i] for i in range(len(input_keys))}
            
        with open("debug_ai.log", "a") as f:
             f.write(f"Mapped Keys: {list(result_map.keys())}\n")
             
        return result_map, None
    except Exception as e:
        with open("debug_ai.log", "a") as f:
             f.write(f"ERROR: {e}\n")
        return {}, str(e)

class DigestSaveRequest(BaseModel):
    title: str
    category: str
    summary_markdown: str
    articles: List[ArticleMetadata]
    analysis_source: Optional[List[KeywordData]] = []
    analysis_digest: Optional[List[KeywordData]] = []

class DigestRead(BaseModel):
    id: int
    title: str
    category: str
    created_at: str

@router.get("/outlets/digests/saved", response_model=List[DigestRead])
async def get_saved_digests(db: Session = Depends(get_db)):
    """Returns list of saved digests."""
    result = await db.execute(select(NewsDigest).order_by(NewsDigest.created_at.desc()))
    digests = result.scalars().all()
    return [
        DigestRead(
            id=d.id,
            title=d.title,
            category=d.category,
            created_at=d.created_at.isoformat()
        ) for d in digests
    ]

class DigestDetail(BaseModel):
    id: int
    title: str
    category: str
    summary_markdown: str
    articles: List[ArticleMetadata]
    analysis_source: Optional[List[KeywordData]] = []
    analysis_digest: Optional[List[KeywordData]] = []
    created_at: str

@router.get("/outlets/digests/{id}", response_model=DigestDetail)
async def get_digest_detail(id: int, db: Session = Depends(get_db)):
    """Returns full details of a saved digest."""
    result = await db.execute(select(NewsDigest).where(NewsDigest.id == id))
    digest = result.scalars().first()
    if not digest:
        raise HTTPException(status_code=404, detail="Digest not found")
    
    articles = []
    if digest.articles_json:
        try:
            raw_articles = json.loads(digest.articles_json)
            articles = [ArticleMetadata(**a) for a in raw_articles]
        except:
            pass
            
    analysis_source = []
    if digest.analysis_source:
        try:
            raw_src = json.loads(digest.analysis_source)
            analysis_source = [KeywordData(**k) for k in raw_src]
        except: pass

    analysis_digest = []
    if digest.analysis_digest:
        try:
            raw_dig = json.loads(digest.analysis_digest)
            analysis_digest = [KeywordData(**k) for k in raw_dig]
        except: pass

    return DigestDetail(
        id=digest.id,
        title=digest.title,
        category=digest.category,
        summary_markdown=digest.summary_markdown,
        articles=articles,
        analysis_source=analysis_source,
        analysis_digest=analysis_digest,
        created_at=digest.created_at.isoformat()
    )



class CityInfoResponse(BaseModel):
    population: str
    description: str
    ruling_party: str
    flag_url: Optional[str] = None
    city_native_name: Optional[str] = None
    city_phonetic_name: Optional[str] = None
    country_flag_url: Optional[str] = None
    country_english: Optional[str] = None
    country_native: Optional[str] = None
    country_phonetic: Optional[str] = None

@router.get("/outlets/city_info", response_model=CityInfoResponse)
async def get_city_info(city: str, country: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Fetches quick city stats using Gemini, with DB caching.
    """
    # 1. Check DB Cache
    stmt = select(CityMetadata).join(Country).where(
        CityMetadata.name == city, 
        (Country.name == country) | (Country.native_name == country)
    )
    stmt = select(CityMetadata).where(CityMetadata.name == city)
    result = await db.execute(stmt)
    cached_city = result.scalars().first()
    
    if cached_city:
        db_country_stmt = select(Country).where(Country.id == cached_city.country_id)
        res_c = await db.execute(db_country_stmt)
        db_country = res_c.scalars().first()
        
        if db_country:
            return CityInfoResponse(
                population=cached_city.population or "Unknown",
                description=cached_city.description or "",
                ruling_party=cached_city.ruling_party or "Unknown",
                flag_url=cached_city.flag_url,
                city_native_name=cached_city.native_name,
                city_phonetic_name=cached_city.phonetic_name,
                country_flag_url=db_country.flag_url,
                country_english=db_country.name,
                country_native=db_country.native_name,
                country_phonetic=db_country.phonetic_name
            )

    # 2. Not cached: Generate
    prompt = f"""
    Provide brief structured info about the city {city}, {country}.
    
    Instructions:
    1. **Country Metadata**: Identify the country's name in English, its Native Language Name (e.g. "România"), and its Phonetic Pronunciation (e.g. "ro-muh-nee-a").
    2. **City Metadata**: Identify the city's Native Name (e.g. "București") and Phonetic Pronunciation.
    3. **Flag**: Find a high-quality Wikimedia URL for the **COUNTRY's Flag** (SVG or PNG).
    4. **City Stats**: Population, 1-sentence description, and Mayor's Party.
    
    Return strictly JSON:
    {{
      "population": "approx X (Year)",
      "description": "1-sentence summary (max 15 words).",
      "ruling_party": "Mayor's Party",
      "flag_url": "URL to City Coat of Arms (optional, can be null)",
      "city_native_name": "București",
      "city_phonetic_name": "/bukuˈreʃtʲ/",
      "country_flag_url": "URL to COUNTRY Flag (Wikimedia SVG preferred)",
      "country_english": "Romania",
      "country_native": "România",
      "country_phonetic": "/ro.mɨˈni.a/" 
    }}
    """
    
    try:
        api_key = current_user.gemini_api_key
        if not api_key: 
            return CityInfoResponse(population="Unknown", description="API Key needed.", ruling_party="Unknown")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')
        response = await model.generate_content_async(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        # 3. Save to DB
        c_eng = data.get('country_english', country)
        stmt_c = select(Country).where(Country.name == c_eng)
        res_c = await db.execute(stmt_c)
        db_country = res_c.scalars().first()
        
        if not db_country:
            db_country = Country(
                name=c_eng,
                native_name=data.get('country_native'),
                phonetic_name=data.get('country_phonetic'),
                flag_url=data.get('country_flag_url')
            )
            db.add(db_country)
            await db.commit()
            await db.refresh(db_country)
        else:
            if not db_country.flag_url and data.get('country_flag_url'):
                db_country.flag_url = data.get('country_flag_url')
                await db.commit()

        db_city = CityMetadata(
            name=city,
            native_name=data.get('city_native_name'),
            phonetic_name=data.get('city_phonetic_name'),
            country_id=db_country.id,
            population=data.get('population'),
            description=data.get('description'),
            ruling_party=data.get('ruling_party'),
            flag_url=data.get('flag_url')
        )
        db.add(db_city)
        await db.commit()

        return CityInfoResponse(**data)

    except Exception as e:
        print(f"City Info Error: {e}")
        return CityInfoResponse(population="Unknown", description=f"Automated data unavailable.", ruling_party="Unknown")

class DigestRequest(BaseModel):
    outlet_ids: List[int]
    category: str
    timeframe: Optional[str] = "24h" # 24h, 3days, 1week

async def robust_fetch(client, url):
    try:
        response = await client.get(url)
        if response.status_code in [301, 302, 307, 308]:
             response = await client.get(response.headers["Location"])
        return response
    except Exception as e:
        print(f"Fetch error {url}: {e}")
        return None

async def smart_scrape_outlet(outlet: NewsOutlet, category: str, timeframe: str = "24h", log_bus: any = None) -> dict:
    """
    Fetches content from an outlet, intelligently navigating to the category page if possible.
    Returns structured article data and raw text for AI.
    """
    
    async def log(msg: str):
         if log_bus:
              await log_bus(msg)
         print(msg) # Keep stdout

    # 1. Determine Target URL
    target_url = outlet.url
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=20, headers=headers) as client:
        # 1. Fetch Homepage
        await log(f"[{outlet.name}] Fetching homepage: {outlet.url}")
        resp = await robust_fetch(client, outlet.url)
        if not resp or resp.status_code != 200:
            code = resp.status_code if resp else "ERR"
            await log(f"Failed to fetch {target_url}: {code}")
            return {"articles": [], "raw_text": ""}
        
        html_content = resp.text
        # Encoding fix
        if resp.encoding and resp.encoding.lower() not in ['utf-8', 'iso-8859-1']:
             try:
                 html_content = resp.content.decode(resp.encoding)
             except:
                 pass # Fallback to .text auto-decode
        
        await log(f"Fetched {len(html_content)} bytes (Encoding: {resp.encoding or 'auto'})")
        
        final_url = outlet.url
        # Limit HTML size to prevent CPU blocking on huge pages
        html = resp.text[:200000] 
        soup = BeautifulSoup(html, 'html.parser')

        # 2. Try to find Category Link (if category is specific)
        if category.lower() not in ["general", "all", "headline"]:
            # naive search for link text or href
            cat_link = None
            term = category.lower()
            # Multi-language mappings for Category navigation
            # EN, RO, ES, FR, DE, IT
            mappings = {
                "politics": [
                    "politics", "politic", "politica", "politique", "politik", 
                    "administratie", "administration", "gobierno", "regierung", "governo"
                ],
                "sports": [
                    "sport", "sports", "deporte", "deportes", "fotbal", "football", "soccer", "futbol"
                ],
                "economy": [
                    "economy", "business", "financial", "economie", "economia", "wirtschaft", "finanzen", 
                    "bani", "money", "dinero", "argent", "geld"
                ],
                "social": [
                    "social", "society", "societate", "sociedad", "société", "gesellschaft", "società", 
                    "community", "comunitate", "comunidad"
                ],
                "culture": [
                    "culture", "cultura", "kultur", "arts", "life", "lifestyle", "monden", "entertainment", 
                    "unterhaltung", "magazin", "magazine"
                ]
            }
            
            # Add the requested category itself as a primary term
            search_terms = mappings.get(term, [])
            if term not in search_terms:
                search_terms.insert(0, term)
            
            for t in search_terms:
                link_tag = soup.find('a', string=lambda text: text and t in text.lower()) or \
                           soup.find('a', href=lambda href: href and t in href.lower())
                if link_tag:
                    href = link_tag.get('href')
                    if href:
                        if href.startswith("/"):
                            # Handle relative URLs
                            import urllib.parse
                            final_url = urllib.parse.urljoin(outlet.url, href)
                        elif href.startswith("http"):
                            final_url = href
                        
                        await log(f"[{outlet.name}] Found category link: {final_url}")
                        cat_resp = await robust_fetch(client, final_url)
                        if cat_resp and cat_resp.status_code == 200:
                            html = cat_resp.text[:200000] # Limit size
                            soup = BeautifulSoup(html, 'html.parser')
                        break

        # 2. Construct Potential Category URLs (Active Discovery)
    # Instead of hoping to find a link on the homepage, we try to guess the category URL.
    # Most RO sites use /politica, /administratie, /sport, etc.
    outlet_url = outlet.url # Use the base URL for construction
    urls_to_scrape = [outlet_url] # Always scrape homepage
    
    # Get relevant keywords for the category
    cat_keywords = mappings.get(category.lower(), [])
    
    # Try to construct specific paths (limit to 2 most likely to save time)
    # e.g. site.ro/politica or site.ro/category/politica
    for kw in cat_keywords[:2]:
        # Clean double slashes
        base = outlet_url.rstrip("/")
        urls_to_scrape.append(f"{base}/{kw}")
        urls_to_scrape.append(f"{base}/stiri/{kw}") # Common pattern
        urls_to_scrape.append(f"{base}/sectiune/{kw}") # Another pattern

    await log(f"DEBUG: Active Scraping for {outlet.name}: {urls_to_scrape}")
    
    # NEW: Dictionary to aggregate metadata by URL
    candidates_map = {} 
    
    # 3. Scrape All Candidates
    import asyncio
    
    # Helper to fetch and parse single URL (placeholder)
    async def fetch_and_parse(target_url):
         try: pass
         except: pass

    # Refactored Loop to process multiple URLs
    combined_content = ""
    
    for i, target_url in enumerate(urls_to_scrape):
        # Limit active scraping to avoid timeouts
        if i > 3: break 
        
        try:
            await log(f"  -> Fetching: {target_url}")
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client: # Reduced timeout for sub-pages
                resp = await client.get(target_url, headers=headers)
                if resp.status_code != 200: continue
                
                # Parse
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Extract Text (Append to combined)
                text = soup.get_text(separator=' ', strip=True)
                # Truncate to avoid exploding token context
                combined_content += f"\n--- SOURCE: {outlet.name} [{target_url}] ---\n{text[:10000]}\n"
                
                # Extract Links (Article Discovery)
                # Reuse the existing article extraction logic?
                # The original code had a big block for "Find Articles".
                # We need to run that block on 'soup'.
                
                # ... (Logic below is the original article extraction, customized for the loop)
                
                # Find all potential links
                links = soup.find_all('a', href=True)
                
                for a in links:
                    href = a['href']
                    raw_title = a.get_text(strip=True)
                    
                    # Normalize URL
                    # Normalize URL
                    if not href.startswith('http'):
                        import urllib.parse
                        # Use target_url (current page) as base if possible, or outlet_url
                        # Actually outlet_url (lines 708) might be just the base.
                        # But href is usually relative to the current path if not absolute.
                        # Safe bet: urljoin on outlet.url is standard for site-wide crawling if we assume href is from root.
                        # But if href is relative "article.html", it should generally be joined with target_url.
                        # HOWEVER, in this loop, target_url changes.
                        # Let's use urljoin(target_url, href) which is most correct for scraping.
                        full_url = urllib.parse.urljoin(target_url, href)
                    else:
                        full_url = href
                    
                    # Deduplication logic (handled via candidates_map later)
                    # if full_url in seen_urls: continue
                    
                    # Basic Validation
                    if len(raw_title) < 5: continue
                    
                    # Content/Allowlist Filter
                    # (Reusing mappings/blacklist logic)
                    is_relevant = False
                    
                    # 1. URL Keywords
                    if any(k in full_url.lower() for k in cat_keywords): is_relevant = True
                    # 2. Title Keywords
                    if any(k in raw_title.lower() for k in cat_keywords): is_relevant = True
                    # 3. Date check (simple slug)
                    if "2026" in full_url or "2025" in full_url: is_relevant = True
                    
                    # Blacklist
                    BLACKLIST_TERMS = [
                        # Meta-pages (Contact, Terms, Privacy)
                        "contact", "terms", "privacy", "cookies", "gdpr", "politica-confidentialitate", "despre", "about", 
                        "redactia", "echipa", "team", "publicitate", "advertising", "cariere", "careers", 
                        "politica-editoriala", "caseta-redactionala", "termeni", "conditii",
                        # Food/Recipes
                        "recipe", "retet", "receta", "recette", "rezept", "ricett", "mancare", "food", "kitchen", "bucatarie", "essen", "cucina",
                        # Horoscope/Astrology
                        "horoscop", "horoscope", "horoskop", "zodiac", "zodiaque", "astrology", "astro",
                        # Gossip/Tabloid
                        "can-can", "cancan", "paparazzi", "gossip", "tabloid", "klatsch", "potins", "monden", "diva", "vedete", "vip",
                        # Games/Quizzes
                        "game", "jocuri", "juego", "jeu", "spiele", "gioc", "quiz", "crossword", "sudoku", "rebus",
                        # Lifestyle/Shopping
                        "lifestyle", "fashion", "moda", "mode", "shop", "magazin-online", "store", "oferte"
                    ]
                    if any(b in full_url.lower() for b in BLACKLIST_TERMS): is_relevant = False
                    
                    if any(b in full_url.lower() for b in BLACKLIST_TERMS): is_relevant = False
                    
                    # DEBUG: Log logic
                    # print(f"DEBUG LINK: {full_url} | Relevant: {is_relevant} | Keywords: {cat_keywords}")

                    # 4. Contextual Relevance (NEW)
                    page_is_category = any(k in target_url.lower() for k in cat_keywords)
                    
                    if not is_relevant and page_is_category:
                         if full_url.count("-") >= 3:
                             is_relevant = True
                         elif re.search(r'/\d+/', full_url):
                             is_relevant = True
                    
                    if is_relevant:
                        print(f"DEBUG: ACCEPTED CANDIDATE {full_url}")
                    else:
                        # Log first few rejections to see why
                        if len(links) < 100 or i < 1: 
                             pass 
                             # print(f"DEBUG: REJECTED {full_url} (PageCat: {page_is_category})")
                        # Attempt Date Parsing - SCAPER ENGINE V2
                        
                        # 1. Use the optimized Engine
                        # We pass the snippet (parent text) as HTML context if needed, or full HTML?
                        # For efficiency in this loop, we might want to just check URL and Link Text first.
                        # BUT scraper_engine is designed for FULL HTML page analysis usually.
                        # Adapted for Discovery Loop:
                        
                        found_date_str = None
                        
                        # A. Check URL (Fastest)
                        url_date_obj = scraper_engine.extract_date_from_url(full_url)
                        if url_date_obj: 
                            found_date_str = url_date_obj.strftime("%Y-%m-%d")
                        
                        # B. Check Link Text / Parent (Heuristic)
                        if not found_date_str:
                             parent_text = a.parent.get_text(separator=' ', strip=True)
                             # Reuse the parse_romanian_date logic which handles various text formats
                             # We try to extract a date substring first? 
                             # Actually `parse_romanian_date` is permissive.
                             # Let's try passing likely candidates.
                             
                             # Heuristic: Check if Parent Text IS a date
                             d = scraper_engine.parse_romanian_date(parent_text[:50]) # Check start
                             if d: found_date_str = d.strftime("%Y-%m-%d")
                             
                        # C. Check Time Tag in Parent
                        if not found_date_str:
                             time_tag = a.find_next("time") or a.find_previous("time") or a.parent.find("time")
                             if time_tag and time_tag.has_attr("datetime"):
                                  t_match = re.search(r'(\d{4}-\d{2}-\d{2})', time_tag["datetime"])
                                  if t_match: found_date_str = t_match.group(1)

                        # Merge/Update Logic
                        existing = candidates_map.get(full_url)
                        
                        if existing:
                              # Update Title if new one is longer (heuristic: longer text = actual headline vs category badge)
                              if len(raw_title) > len(existing.title):
                                   existing.title = raw_title
                              
                              # Update Date if missing
                              if not existing.date_str and found_date_str:
                                  existing.date_str = found_date_str
                        else:
                             # Create New
                             art = ArticleMetadata(
                                 source=outlet.name,
                                 title=raw_title,
                                 url=full_url,
                                 date_str=found_date_str
                             )
                             candidates_map[full_url] = art
                        
        except Exception as e:
            await log(f"Error scraping {target_url}: {e}")
            continue

    # Finalize: Convert values to list
    all_extracted_articles = list(candidates_map.values())

    # Return aggregated result (matching the expected dict structure)
    return {
        "text": combined_content,
        "articles": all_extracted_articles
    }

async def generate_keyword_analysis(text: str, category: str, current_user: User) -> List[KeywordData]:
    """Reusable function to analyze text and extract keywords/sentiment."""
    if not text:
        return []

    api_key = current_user.gemini_api_key
    if not api_key: return []
    genai.configure(api_key=api_key)

    # Truncate to avoid context limits if very large
    text_sample = text[:30000]

    prompt = f"""
    Analyze the following news text specifically for the category '{category}'. IGNORE topics unrelated to {category}.
    Extract the Top 100 most significant keywords/terms that are strictly grounded in the text.
    
    Instructions:
    1. **Strict Category Relevance**: Only include terms directly related to "{category}". For example, if category is "Politics", exclude "Football" or "Celebrity Gossip" unless directly political.
    2. **Entities**: Identify specific Persons, Locations, Organizations, and Events (e.g., "Mayor Boc", "Cluj-Napoca", "Untold Festival").
    3. **Concepts**: Identify key themes or objects (e.g., "Budget", "Traffic", "Pollution").
    4. **Filter**: EXCLUDE generic stopwords (and, the, if, but, etc.) and generic news terms (reporter, news, article).
    5. **Metadata**:
        - **Importance**: Score 1-100 based on relevance/frequency.
        - **Type**: Person, Location, Organization, Concept, Event, Object.
        - **Sentiment**: Detect the context/emotion associated with this specific term in the text. 
          Examples: Positive, Negative, Balanced, Accusatory, Praise, Fearful, Controversial.
    
    Return strictly a JSON list:
    [
        {{ "word": "Emil Boc", "importance": 95, "type": "Person", "sentiment": "Positive" }},
        {{ "word": "Traffic", "importance": 80, "type": "Concept", "sentiment": "Negative" }}
    ]

    Text to Analyze:
    {text_sample}
    """

    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = await model.generate_content_async(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        results = []
        for item in data:
            results.append(KeywordData(
                word=item.get('word', 'Unknown'),
                importance=int(item.get('importance', 50)),
                type=item.get('type', 'Concept'),
                sentiment=item.get('sentiment', 'Neutral')
            ))
        
        # Sort by importance descending
        results.sort(key=lambda x: x.importance, reverse=True)
        return results
    except ResourceExhausted:
         raise HTTPException(status_code=429, detail="AI Quota Exceeded. Please try again later.")
    except Exception as e:
        print(f"Analysis Failed: {e}")
        return []

# --- Digest Management Models ---
class DigestCreate(BaseModel):
    title: str
    category: str
    summary_markdown: str
    articles: List[Dict[str, Any]] # Will be serialized to JSON
    analysis_source: Optional[List[Dict[str, Any]]] = None # Will be serialized to JSON
    analysis_digest: Optional[List[Dict[str, Any]]] = None

class DigestRead(DigestCreate):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- Digest Endpoints ---

@router.post("/digests", response_model=DigestRead)
async def save_digest(
    digest: DigestCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a generated digest to the database."""
    print(f"DEBUG: Saving digest '{digest.title}' for user {current_user.id}")
    import json
    
    db_digest = NewsDigest(
        user_id=current_user.id,
        title=digest.title,
        category=digest.category,
        summary_markdown=digest.summary_markdown,
        articles_json=json.dumps(digest.articles),
        analysis_source=json.dumps(digest.analysis_source) if digest.analysis_source else None,
        analysis_digest=json.dumps(digest.analysis_digest) if digest.analysis_digest else None
    )
    db.add(db_digest)
    await db.commit()
    await db.refresh(db_digest)
    
    # Clean return (deserialize for response)
    return DigestRead(
        id=db_digest.id,
        title=db_digest.title,
        category=db_digest.category,
        summary_markdown=db_digest.summary_markdown,
        articles=json.loads(db_digest.articles_json),
        analysis_source=json.loads(db_digest.analysis_source) if db_digest.analysis_source else [],
        created_at=db_digest.created_at
    )

@router.get("/digests", response_model=List[DigestRead])
async def list_digests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all saved digests for the current user."""
    import json
    stmt = select(NewsDigest).where(NewsDigest.user_id == current_user.id).order_by(NewsDigest.created_at.desc())
    result = await db.execute(stmt)
    digests = result.scalars().all()
    
    # Manual mapping to handle JSON deserialization
    return [
        DigestRead(
            id=d.id,
            title=d.title,
            category=d.category,
            summary_markdown=d.summary_markdown,
            articles=json.loads(d.articles_json) if d.articles_json else [],
            analysis_source=json.loads(d.analysis_source) if d.analysis_source else [],
            created_at=d.created_at
        ) for d in digests
    ]

@router.delete("/digests/{digest_id}")
async def delete_digest(
    digest_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a saved digest."""
    stmt = select(NewsDigest).where(NewsDigest.id == digest_id, NewsDigest.user_id == current_user.id)
    result = await db.execute(stmt)
    digest = result.scalar_one_or_none()
    
    if not digest:
        raise HTTPException(status_code=404, detail="Digest not found")
        
    await db.delete(digest)
    await db.commit()
    return {"status": "success", "message": "Digest deleted"}

# --- AI RELEVANCE CHECK ---
async def verify_relevance_with_ai(title: str, url: str, category: str, api_key: str) -> bool:
    """
    Uses Gemini to strictly verify if an article is relevant to the category.
    Returns True if relevant, False otherwise.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
        Analyze if the following news article is relevant to the category '{category}'.
        Title: {title}
        URL: {url}
        
        Rules:
        1. Context matters. "Water cutoff" is NOT Politics. "Mayor announces water cutoff" IS Politics/Administration.
        2. "Traffic accident" is NOT Politics/Administration.
        3. If it is a generic utility announcement, specific crime report (robbery), or gossip, return FALSE.
        4. If it is about city council, mayor, public spending, laws, healthcare policy, education policy, infrastructure projects, return TRUE.
        
        Respond with exactly ONE word: TRUE or FALSE.
        """
        
        response = await model.generate_content_async(prompt)
        ans = response.text.strip().upper()
        return "TRUE" in ans
    except Exception as e:
        print(f"AI Verification Failed: {e}")
        return True # Fail open to avoid dropping potentially good articles if API fails

# Removed OLD extract_date_with_ai (Moved to scraper_engine)

from fastapi.responses import StreamingResponse
import json
import asyncio

@router.post("/outlets/digest/stream")
async def generate_digest_stream(req: DigestRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Streams log updates and final result as NDJSON.
    """
    
    async def process_stream():
        # Queue for cross-task communication
        stream_queue = asyncio.Queue()
        
        # Callback wrapper to put logs into queue
        async def queue_logger(msg: str):
            await stream_queue.put({"type": "log", "message": msg})

        yield json.dumps({"type": "log", "message": "Initializing Secure Pipeline..."}) + "\n"
        
        # 1. Fetch Outlets
        stmt = select(NewsOutlet).where(NewsOutlet.id.in_(req.outlet_ids))
        result = await db.execute(stmt)
        outlets = result.scalars().all()
        
        if not outlets:
             yield json.dumps({"type": "error", "message": "No outlets found"}) + "\n"
             return

        yield json.dumps({"type": "log", "message": f"Targeting {len(outlets)} sources..."}) + "\n"
        
        # WORKER FUNCTION
        async def scraper_worker():
            try:
                all_raw_articles = []
                for outlet in outlets:
                     # Pass queue_logger which is Awaitable (not a generator)
                     res = await smart_scrape_outlet(outlet, req.category, req.timeframe, log_bus=queue_logger)
                     
                     if res.get("articles"):
                          all_raw_articles.extend(res["articles"])
                          await stream_queue.put({"type": "log", "message": f"Found {len(res['articles'])} articles from {outlet.name}"})
                
                # Signal phase change or return data
                print(f"DEBUG: WORKER FINISHED. Sending {len(all_raw_articles)} articles.")
                await stream_queue.put({"type": "data", "articles": all_raw_articles})
            except Exception as e:
                print(f"DEBUG: WORKER ERROR: {e}")
                await stream_queue.put({"type": "error", "message": str(e)})
            finally:
                await stream_queue.put(None) # Sentinel

        # Start Worker
        task = asyncio.create_task(scraper_worker())
        
        all_articles = []
        
        # Consumer Loop
        while True:
            item = await stream_queue.get()
            if item is None:
                break
            
            if item["type"] == "log":
                yield json.dumps(item) + "\n"
            elif item["type"] == "error":
                yield json.dumps(item) + "\n"
            elif item["type"] == "data":
                all_articles = item["articles"]
                print(f"DEBUG: CONSUMER RECEIVED {len(all_articles)} ARTICLES")
                yield json.dumps({"type": "log", "message": f"Processing {len(all_articles)} raw items..."}) + "\n"
        
        print(f"DEBUG: CONSUMER EXITED LOOP. Total Articles: {len(all_articles)}")
        
        await task # Ensure clean exit check

        # ... (Rest of logic: Scoring, Rescue, AI)
        
        # COPY OF THE DIGEST LOGIC (REFACTORED)
        
        yield json.dumps({"type": "log", "message": "Ranking & Scoring Articles..."}) + "\n"
        
        # 0. Timeframe Calculation
        from datetime import datetime, timedelta
        now = datetime.now()
        cutoff_date = now - timedelta(days=1) # Default 24h
        if req.timeframe == "3days":
            cutoff_date = now - timedelta(days=3)
        elif req.timeframe == "1week":
            cutoff_date = now - timedelta(days=7)
            
        yield json.dumps({"type": "log", "message": f"Timeframe: {req.timeframe} (Cutoff: {cutoff_date.date()})"}) + "\n"

        candidates_for_ai = []
        filtered_articles = [] # Final list
        analysis_source = [] # We skip detailed keyword analysis for stream to save time/quota

        total_scraped = len(all_articles)
        yield json.dumps({"type": "log", "message": f"Scoring {total_scraped} raw articles..."}) + "\n"
        
        # Helper lists
        BLOCKED_DOMAINS = ["google.com", "apple.com", "youronlinechoices", "facebook.com", "twitter.com", "instagram.com", "tiktok.com", "youtube.com"]
        SUSPICIOUS_TERMS = [
            "recipe", "retet", "receta", "recette", "rezept", "ricett", "mancare", "food", "kitchen", "bucatarie", "essen", "cucina",
            "horoscop", "horoscope", "horoskop", "zodiac", "zodiaque", "astrology",
            "can-can", "cancan", "paparazzi", "gossip", "tabloid", "klatsch", "potins", "cookie", "gdpr", "privacy", "termeni", "conditii"
        ]
        NOISE_TERMS = [
            "apa calda", "apa rece", "intrerupere", "avarie", "curent", "electricitate",
            "trafic", "restrictii", "accident", "incendiu", "minor", "program",
            "meteo", "vremea", "prognoza", "cod galben", "cod portocaliu"
        ]

        # DEDUPLICATION SETS
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in all_articles:
             # URL Normalization for Dedupe
             norm_url = article.url.split("?")[0].rstrip("/")
             if norm_url in seen_urls: continue
             
             # Title Dedupe (Simple lowercasing)
             norm_title = article.title.lower().strip()
             if norm_title in seen_titles: continue
             
             seen_urls.add(norm_url)
             seen_titles.add(norm_title)
             
             # SPAM BLOCK
             if any(d in article.url for d in BLOCKED_DOMAINS): continue
             # CATEGORY BLOCK
             if "/category/" in article.url or "/page/" in article.url or "/tag/" in article.url or "/eticheta/" in article.url or "/author/" in article.url or "/autor/" in article.url: continue
            
             # Find Source Outlet for strict filtering
             source_outlet = next((o for o in outlets if o.name == article.source), None)
             if source_outlet and "#" in article.url and article.url.split("#")[0] == source_outlet.url: continue

             # SCORING
             topic_score = 0
             title_lower = article.title.lower()
             url_lower = article.url.lower()
             
             # ... (reusing existing scoring logic) ...
             # Simple heuristic for category matching
             if req.category.lower() in title_lower or req.category.lower() in url_lower:
                 topic_score += 30
                 
             # Contextual URL Boost
             cat_stem = req.category.lower()[:4] 
             if f"/{cat_stem}" in url_lower:
                  topic_score += 20
                  
             # Generic "Admin" boost
             if req.category in ["Politics", "Admin"]:
                 if any(k in title_lower for k in ["primar", "consiliu", "presedinte", "ministru", "guvern", "parlament"]):
                     topic_score += 40
                 elif any(k in title_lower for k in ["scandal", "acuzatii", "demisie", "alegeri"]):
                     topic_score += 50
                 elif "sibi" in title_lower:
                     topic_score += 10
             
             # EXPLICIT PENALTIES
             if req.category.lower() not in ["local", "general", "all"]:
                  if any(n in title_lower for n in NOISE_TERMS):
                      topic_score -= 50

             # Penalize Off-Topic
             if any(term in title_lower for term in SUSPICIOUS_TERMS) or any(term in url_lower for term in SUSPICIOUS_TERMS):
                  topic_score -= 100
             
             # Date Logic
             date_score = 0
             is_within_timeframe = False
             
             if article.date_str:
                  try:
                      d_obj = datetime.strptime(article.date_str, "%Y-%m-%d")
                      if d_obj >= cutoff_date:
                          date_score = 30
                          is_within_timeframe = True
                  except: pass
            
             total_score = topic_score 
             
             # Date Bonus/Penalty
             if is_within_timeframe:
                 date_score = 30
                 total_score += date_score
             elif article.date_str:
                 # INVALID DATE (Old) -> Strict Rejection as requested
                 date_score = 0
                 total_score = 0 
             else:
                 # Undated -> Allow (maybe?) or Strict? 
                 # User said "check the dates... mark ones outside with red".
                 # Implies strictly checking KNOWN dates.
                 # If date is unknown, we can't be sure it's outside.
                 # Let's keep undated as "neutral" (0 bonus) but ALLOWED if topic is high to avoid empty report again.
                 date_score = 0
             
             article.relevance_score = int(total_score)
             # Inject Metadata for Frontend
             article.scores = {
                 "topic": topic_score, 
                 "date": date_score, 
                 "is_fresh": is_within_timeframe,
                 "is_old": (article.date_str and not is_within_timeframe)
             }
             
             # Filter thresholds (Keep permissive but transparent)
             # User wants EVERYTHING in the table
             unique_articles.append(article)

        # AI TITLE PRE-FILTER
        # Filter logic:
        # 1. Take top candidates (e.g. all unique_articles which passed heuristic, or top N)
        # 2. Batch verify titles
        # 3. Filter out rejections OR Penalize
        
        candidates_to_verify = unique_articles # For now verify all that passed basic filters
        
        if candidates_to_verify and current_user.gemini_api_key:
             yield json.dumps({"type": "log", "message": f"🤖 AI Pre-Filtering {len(candidates_to_verify)} titles..."}) + "\n"
             
             # Prepare batch (Assign IDs for stability)
             titles_map = {i: art.title for i, art in enumerate(candidates_to_verify)}
             
             # Chunking (Gemini has limits, maybe 50 at a time)
             chunk_size = 50
             verified_results = {}
             
             title_ids = list(titles_map.keys())
             for i in range(0, len(title_ids), chunk_size):
                 chunk_ids = title_ids[i:i+chunk_size]
                 chunk_map = {k: titles_map[k] for k in chunk_ids}
                 
                 res, err = await batch_verify_titles_debug(chunk_map, POLITICS_OPERATIONAL_DEFINITION, current_user.gemini_api_key)
                 if err:
                      yield json.dumps({"type": "log", "message": f"⚠️ Batch AI Error: {err}"}) + "\n"
                 
                 verified_results.update(res)
             
             # Apply verdicts
             filtered_articles = []
             # Apply verdicts - NO FILTERING, just tagging
             for i, art in enumerate(candidates_to_verify):
                 # Result keys are strings in JSON
                 verdict = verified_results.get(str(i), verified_results.get(i)) 
                 
                 if verdict is True:
                     # CONFIRMED POLITICS
                     art.relevance_score += 20 # Bonus
                     art.ai_verdict = "PASS"
                 elif verdict is False:
                     # CONFIRMED NOT POLITICS
                     art.relevance_score -= 10 # Penalty but keep
                     art.ai_verdict = "FAIL"
                 else:
                     # Error/Missing
                     art.ai_verdict = "UNKNOWN"
        else:
             yield json.dumps({"type": "log", "message": "⚠️ Skipping AI Filter (No API Key or No Candidates)"}) + "\n"
        
        # ALL articles go to final list now
        filtered_articles = unique_articles

        # FINAL COMPILE
        yield json.dumps({"type": "log", "message": "Compiling HTML Digest..."}) + "\n"
        
        # Create HTML Table
        
        start_str = cutoff_date.strftime("%b %d")
        end_str = now.strftime("%b %d")
        period_label = f"{start_str} - {end_str}"
        table_html = f"<h1 style='color: #e2e8f0; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 20px;'>Deep Analysis: {req.category} <span style='font-size:0.6em; color:#94a3b8;'>({period_label})</span></h1>"
        
        # Grouping
        outlet_articles_map = {o.name: [] for o in outlets}
        for article in filtered_articles:
            if article.source in outlet_articles_map:
                outlet_articles_map[article.source].append(article)
        
        # Sort Outlets
        sorted_outlets = sorted(outlets, key=lambda o: max([a.relevance_score for a in outlet_articles_map[o.name]] or [0]), reverse=True)
        
        for outlet in sorted_outlets:
            arts = outlet_articles_map.get(outlet.name, [])
            if not arts: continue
            
            # Sort Articles
            arts.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Outlet Header
            table_html += f"""
            <div style="margin-top: 32px; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 8px;">
                <h3 style="margin: 0; font-size: 1.4rem; color: #f8fafc;">
                    <a href="{outlet.url}" target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: bold;">{outlet.name}</a>
                    <span style="color: #94a3b8; font-size: 1rem; font-weight: normal; margin-left: 10px;">({outlet.city})</span>
                    <span class="scraper-debug-trigger" data-url="{outlet.url}" style="cursor: pointer; font-size: 0.8em; margin-left: 8px; vertical-align: middle; opacity: 0.5;" title="Debug Scraper Rules">🔧</span>
                </h3>
            </div>
            """
                
            # Table Header
            table_html += """
            <table style="width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.95rem; margin-bottom: 24px; border: 1px solid #334155; border-radius: 6px; overflow: hidden;">
                <thead style="background-color: #1e293b; color: #e2e8f0;">
                    <tr>
                        <th style="padding: 12px 16px; text-align: left; font-weight: 600; border-bottom: 1px solid #334155;">Assess</th>
                        <th style="padding: 12px 16px; text-align: center; font-weight: 600; border-bottom: 1px solid #334155;">AI Check</th>
                        <th style="padding: 12px 16px; text-align: center; font-weight: 600; border-bottom: 1px solid #334155;">Date</th>
                        <th style="padding: 12px 16px; text-align: center; font-weight: 600; border-bottom: 1px solid #334155;">Topic</th>
                        <th style="padding: 12px 16px; text-align: left; font-weight: 600; border-bottom: 1px solid #334155;">Article</th>
                    </tr>
                </thead>
                <tbody style="background-color: #0f172a;">
            """
            
            for art in arts:
                s = art.scores
                
                topic_display = f"{s.get('topic', 0)}"
                
                date_color = "#4ade80" if art.relevance_score > 0 else "#7f1d1d"
                date_display = f"<span style='color: {date_color}; font-weight: bold;'>{art.date_str}</span>" if art.date_str else f"<span style='color: #7f1d1d;'>N/A</span>"
                
                # Add Debug Trigger to Date
                date_display += f"""<span class="scraper-debug-trigger" data-url="{art.url}" style="cursor: pointer; margin-left: 6px; font-size: 0.8em; opacity: 0.6;" title="Debug Date Extraction">🔧</span>"""

                # Score Styling
                if art.relevance_score > 80:
                    score_bg = "#052e16" 
                    score_text = "#4ade80" 
                    score_border = "#15803d"
                elif art.relevance_score > 50:
                    score_bg = "#422006" 
                    score_text = "#facc15" 
                    score_border = "#a16207"
                else:
                    score_bg = "#450a0a" 
                    score_text = "#f87171" 
                    score_border = "#b91c1c"
                
                # Escape for HTML attributes
                safe_url = html.escape(art.url)
                safe_title = html.escape(art.title)

                score_badge = f"""
                <button class="politics-assess-trigger" 
                        data-url="{safe_url}" 
                        data-title="{safe_title}"
                        style="
                            display: inline-flex; align-items: center; gap: 4px;
                            background-color: #1e293b; color: #94a3b8; 
                            border: 1px solid #334155; padding: 4px 8px; 
                            border-radius: 6px; font-weight: bold; font-size: 0.7rem; 
                            cursor: pointer; transition: all 0.2s;
                        "
                        onmouseover="this.style.backgroundColor='#334155'; this.style.color='white';"
                        onmouseout="this.style.backgroundColor='#1e293b'; this.style.color='#94a3b8';"
                >
                    🤖 Assess
                </button>
                <div class="politics-verdict" data-url="{art.url}" style="margin-top: 4px; font-size: 0.7rem; display: none;"></div>
                """
                
                title_color = "#f1f5f9"
                
                # AI Status Column Logic
                verdict_icon = "❓"
                verdict_color = "#94a3b8"
                verdict_tooltip = "Not Verified"
                
                if hasattr(art, 'ai_verdict'):
                    if art.ai_verdict == "PASS":
                        verdict_icon = "✅"
                        verdict_color = "#4ade80"
                        verdict_tooltip = "AI Confirmed: Fits Politics Definition"
                    elif art.ai_verdict == "FAIL":
                        verdict_icon = "⚠️"
                        verdict_color = "#fbbf24"
                        verdict_tooltip = "AI Warning: Likely Off-Topic (but shown per request)"
                
                ai_check_html = f"""
                <div style="text-align: center; font-size: 1.2em; cursor: help;" title="{verdict_tooltip}">
                    {verdict_icon}
                </div>
                """
                
                table_html += f"""
                    <tr style="border-bottom: 1px solid #1e293b;">
                        <td style="padding: 10px 16px; border-bottom: 1px solid #1e293b;">{score_badge}</td>
                        <td style="padding: 10px 16px; border-bottom: 1px solid #1e293b;">{ai_check_html}</td>
                        <td style="padding: 10px 16px; text-align: center; border-bottom: 1px solid #1e293b; font-size: 0.9rem; color: #cbd5e1;">{date_display}</td>
                        <td style="padding: 10px 16px; text-align: center; border-bottom: 1px solid #1e293b; font-size: 0.9rem; color: #cbd5e1;">{topic_display}</td>
                        <td style="padding: 10px 16px; border-bottom: 1px solid #1e293b;">
                            <a href="{safe_url}" target="_blank" style="display: block; color: {title_color}; text-decoration: none; font-size: 1rem; font-weight: 500; line-height: 1.4; transition: color 0.2s;">
                                <span style="border-bottom: 1px dotted #94a3b8;">{safe_title}</span> <span style="font-size: 0.8em; text-decoration: none;">🔗</span>
                            </a>
                        </td>
                    </tr>
                """
            
            table_html += "</tbody></table>"
            
        final_result = {
            "digest": table_html,
            "articles": [a.dict() for a in filtered_articles],
            "analysis_source": analysis_source or [], 
            "analysis_digest": [],
            "category": req.category
        }
        
        yield json.dumps({"type": "result", "payload": final_result}) + "\n"

    # yield json.dumps({"type": "log", "message": "Processing..."}) + "\n" # OLD PLACEHOLDER
    
    return StreamingResponse(process_stream(), media_type="application/x-ndjson")

# Existing Endpoint (unchanged for backward compat)
@router.post("/outlets/digest", response_model=DigestResponse)
async def generate_digest(req: DigestRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    (Legacy/Simple) Aggregates news.
    """
    # 1. Fetch Outlets
    stmt = select(NewsOutlet).where(NewsOutlet.id.in_(req.outlet_ids))
    result = await db.execute(stmt)

    outlets = result.scalars().all()
    
    if not outlets:
         raise HTTPException(status_code=400, detail="No valid outlets selected")

    # 2. Parallel Smart Scrape
    print(f"Digest: Smart scraping {len(outlets)} outlets for '{req.category}' within {req.timeframe}...")
    scrape_tasks = [smart_scrape_outlet(o, req.category, req.timeframe) for o in outlets]
    scrape_results = await asyncio.gather(*scrape_tasks)
    
    combined_text = "\\n".join([r['text'] for r in scrape_results])
    all_articles = []
    for r in scrape_results:
        all_articles.extend(r['articles'])
    
    # 3. Parallel Processing: Source Analysis & Table Generation
    print("Digest: Starting Source Analysis & Table Generation...")
    
    # Task A: Analyze SOURCES (Raw Content) - Keep this for keyword extraction
    async def analyze_sources_task():
        # Inject Category for relevance
        results = await generate_keyword_analysis(combined_text, req.category, current_user)
        
        # Post-process: Map to specific source URLs purely by simplified text matching
        final_results = []
        for kw in results:
            kw.source_urls = []
            
            # 1. Strict Source Verification
            # Keyword must appear in the article title or match the URL content
            match_found = False
            for article in all_articles:
                # Use title and URL for matching since we don't store full text in metadata object
                if kw.word.lower() in article.title.lower() or kw.word.lower() in article.url.lower():
                     if article.url and article.url not in kw.source_urls:
                         kw.source_urls.append(article.url)
                         match_found = True
            
            # 2. Ghost Filter: If no sources contain this keyword, discard it.
            if match_found:
                 final_results.append(kw)
        
        return final_results

    # Execute Analysis Task
    t_source_analysis = asyncio.create_task(analyze_sources_task())
    analysis_source = await t_source_analysis
    
    # --- TABLE GENERATION (Replaces Digest) ---
    
    # 1. Relevance Scoring (Strict 3-Factor)
    SUSPICIOUS_TERMS = [
        "recipe", "retet", "receta", "recette", "rezept", "ricett", "mancare", "food", "kitchen", "bucatarie", "essen", "cucina",
        "horoscop", "horoscope", "horoskop", "zodiac", "zodiaque", "astrology",
        "can-can", "cancan", "paparazzi", "gossip", "tabloid", "klatsch", "potins", "cookie", "gdpr", "privacy", "termeni", "conditii"
    ]
    BLOCKED_DOMAINS = ["google.com", "apple.com", "youronlinechoices", "facebook.com", "twitter.com", "instagram.com", "tiktok.com", "youtube.com"]

    # Map articles to outlets for the table
    outlet_articles_map = {o.name: [] for o in outlets}

    # 0. Timeframe Calculation
    from datetime import datetime, timedelta
    now = datetime.now()
    cutoff_date = now - timedelta(days=1) # Default 24h
    if req.timeframe == "3days":
        cutoff_date = now - timedelta(days=3)
    elif req.timeframe == "1week":
        cutoff_date = now - timedelta(days=7)
    
    # Reset cutoff to start of day? No, rolling window is fine or user preference.
    # Let's clean it to be strictly comparative.


    # 1. Processing & Scoring
    filtered_articles = [] # Initialize list

    
    # 1. Processing & Scoring
    filtered_articles = [] # Final list
    candidates_for_ai = [] # Tuples of (article, task)
    
    # Pre-scoring loop
    for article in all_articles:
        # SPAM BLOCK
        # SPAM BLOCK
        if any(d in article.url for d in BLOCKED_DOMAINS): continue
        # CATEGORY BLOCK
        if "/category/" in article.url or "/page/" in article.url or "/tag/" in article.url or "/eticheta/" in article.url or "/author/" in article.url or "/autor/" in article.url: continue
        
        # Find Source Outlet for strict filtering
        source_outlet = next((o for o in outlets if o.name == article.source), None)
        
        # Filter out anchor links on same page
        if source_outlet and "#" in article.url and article.url.split("#")[0] == source_outlet.url: continue

        # SCORING
        # FACTOR 1: TOPIC (MAX ~90 pts)
        topic_score = 0
        title_lower = article.title.lower()
        url_lower = article.url.lower()
        
        # Check against keywords extracted from Source/Digest Analysis
        if analysis_source:
             for kw in analysis_source:
                  if kw.word.lower() in title_lower or kw.word.lower() in url_lower:
                       topic_score += 40 # Boost for matching analyzed keywords
                       break # Cap at one match to avoid inflation
        
        # Simple heuristic for category matching if keywords fail
        if req.category.lower() in title_lower or req.category.lower() in url_lower:
            topic_score += 30
            
        # Contextual URL Boost (e.g. /politica/ in URL)
        cat_stem = req.category.lower()[:4] # "poli" for politics
        if f"/{cat_stem}" in url_lower:
             topic_score += 20

        # --- NOISE PENALTY ---
        NOISE_TERMS = [
            "apa calda", "apa rece", "intrerupere", "avarie", "curent", "electricitate",
            "trafic", "restrictii", "accident", "incendiu", "minor", "program",
            "meteo", "vremea", "prognoza", "cod galben", "cod portocaliu"
        ]
        if req.category.lower() not in ["local", "general", "all"]:
             if any(n in title_lower for n in NOISE_TERMS):
                 topic_score -= 50 # Heavier penalty

        # Penalize Off-Topic
        if any(term in title_lower for term in SUSPICIOUS_TERMS) or any(term in url_lower for term in SUSPICIOUS_TERMS):
             topic_score -= 100 

        # FACTOR 2: GEOGRAPHY (35 pts) - Kept for internal logic/sorting, removed from UI
        geo_score = 0
        target_cities = {o.city.lower() for o in outlets}
        if any(c in title_lower or c in url_lower for c in target_cities):
             geo_score = 35
        else:
             source_outlet = next((o for o in outlets if o.name == article.source), None)
             if source_outlet: geo_score = 35 

        # FACTOR 3: DATE (Strict Filtering)
        date_score = 0
        is_within_timeframe = False
        
        if article.date_str:
             try:
                 # Auto-detect format (YYYY-MM-DD usually)
                 # fast parse
                 d_obj = datetime.strptime(article.date_str, "%Y-%m-%d")
                 if d_obj >= cutoff_date:
                     date_score = 30
                     is_within_timeframe = True
             except:
                 pass
        
        # User Rule: 
        # "if the date of an article falls within the digest time frame then multiply the topic-score by 1"
        # "If the date falls outside of the time-frame or the date is N/A the the score is 0"
        
        if is_within_timeframe:
             # Valid Date
             total_score = topic_score + date_score
        else:
             # Invalid / Old Date
             total_score = 0

        article.relevance_score = int(total_score)
        article.scores = {"topic": topic_score, "geo": geo_score, "date": date_score}
        
        # LOGIC:
        # 1. Must have valid Date (score >= 30) AND Topic Score >= 20 (lowered to allow AI to decide)
        # 2. If it passes AI check, it gets in.
        
        if date_score >= 30 and topic_score >= 20:
            # Candidate for AI
            candidates_for_ai.append(article)
        elif topic_score >= 20 and date_score < 30:
            # AI DATE RESCUE MISSION
            # Relevant topic, but missing date. Try to rescue it.
            # (Runs for any topic >= 20, even if date check failed)
            try:
                 print(f"Rescuing Date for: {article.title}")
                 async with httpx.AsyncClient() as client:
                      resp = await client.get(article.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                      if resp.status_code == 200:
                           rescued_date = await scraper_engine.extract_date_with_ai(resp.text, article.url, current_user.gemini_api_key)
                           
                           if rescued_date and "429" in rescued_date:
                                # RATE LIMIT HIT
                                print(f"  -> Rate Limit 429: {rescued_date}")
                                analysis_source.append(KeywordData(word="RATE_LIMIT", importance=1, type="System:RateLimit", sentiment="Warning")) # Hack to signal frontend
                           elif rescued_date:
                                # Validate Rescued Date against Cutoff!
                                try:
                                    d_obj = datetime.strptime(rescued_date, "%Y-%m-%d")
                                    if d_obj >= cutoff_date:
                                         print(f"  -> Rescued Valid Date: {rescued_date}")
                                         article.date_str = rescued_date
                                         # Bump Score
                                         article.relevance_score = topic_score + 30 + 20 # Score reset: Topic + Date(30) + RescueBonus(20)
                                         article.scores['date'] = 30
                                         # Now it qualifies for AI verification or basic inclusion
                                         candidates_for_ai.append(article)
                                    else:
                                         print(f"  -> Rescued OLD Date: {rescued_date} (Too Old)")
                                         article.date_str = rescued_date
                                         article.relevance_score = 0 # STRICT: It's old, so 0 score.
                                         filtered_articles.append(article) # Include as "Old" reference
                                except:
                                     pass
                           else:
                                # Failed Rescue, still include as fallback if topic is valid
                                article.relevance_score = 0 # No confirmed date = 0 score
                                filtered_articles.append(article)
                      else:
                           article.relevance_score = 0
                           filtered_articles.append(article)
            except Exception as e:
                 print(f"Rescue Failed: {e}")
                 # Fallback
                 article.relevance_score = 0 
                 filtered_articles.append(article)

        elif article.relevance_score > 30 and topic_score > 10:
             # Fallback for "Decently High Score" but maybe weak on specific keywords
             filtered_articles.append(article)

    # Batch AI Verification
    if candidates_for_ai:
        print(f"Verifying {len(candidates_for_ai)} articles with AI...")
        tasks = []
        for art in candidates_for_ai:
             tasks.append(verify_relevance_with_ai(art.title, art.url, req.category, current_user.gemini_api_key))
        
        results = await asyncio.gather(*tasks)
        
        for art, is_relevant in zip(candidates_for_ai, results):
            if is_relevant:
                # PASSED AI
                filtered_articles.append(art)
            else:
                print(f"AI Rejected: {art.title}")
    
    # Mapping
    outlet_articles_map = {o.name: [] for o in outlets}
    for article in filtered_articles:
        if article.source in outlet_articles_map:
            outlet_articles_map[article.source].append(article)

    # 2. Build HTML Table (Dark Mode Optimized)
    # REMOVED LOC COLUMN
    
    start_str = cutoff_date.strftime("%b %d")
    end_str = now.strftime("%b %d")
    period_label = f"{start_str} - {end_str}"
    
    table_html = f"<h1 style='color: #e2e8f0; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 20px;'>Deep Analysis: {req.category} <span style='font-size:0.6em; color:#94a3b8;'>({period_label})</span></h1>"
    
    for outlet in outlets:
        articles = outlet_articles_map.get(outlet.name, [])
        if not articles: continue # Skip empty outlets to save space
        
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Outlet Header
        table_html += f"""
        <div style="margin-top: 32px; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 8px;">
            <h3 style="margin: 0; font-size: 1.4rem; color: #f8fafc;">
                <a href="{outlet.url}" target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: bold;">{outlet.name}</a>
                <span style="color: #94a3b8; font-size: 1rem; font-weight: normal; margin-left: 10px;">({outlet.city})</span>
            </h3>
        </div>
        """
            
        # Table Header (NO LOC)
        table_html += """
        <table style="width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.95rem; margin-bottom: 24px; border: 1px solid #334155; border-radius: 6px; overflow: hidden;">
            <thead style="background-color: #1e293b; color: #e2e8f0;">
                <tr>
                    <th style="padding: 12px 16px; text-align: left; font-weight: 600; border-bottom: 1px solid #334155;">Score</th>
                    <th style="padding: 12px 16px; text-align: center; font-weight: 600; border-bottom: 1px solid #334155;">Date</th>
                    <th style="padding: 12px 16px; text-align: center; font-weight: 600; border-bottom: 1px solid #334155;">Topic</th>
                    <th style="padding: 12px 16px; text-align: left; font-weight: 600; border-bottom: 1px solid #334155;">Article</th>
                </tr>
            </thead>
            <tbody style="background-color: #0f172a;">
        """
        
        for art in articles:
            s = art.scores
            
            # Icons & Text
            date_icon = "✅" if s['date'] >= 30 and art.relevance_score > 0 else "⚠️"
            
            date_color = "#4ade80" if art.relevance_score > 0 else "#7f1d1d" # Green vs Bordeaux Red
            date_display = f"<span style='color: {date_color}; font-weight: bold;'>{art.date_str}</span>" if art.date_str else f"<span style='color: #7f1d1d;'>N/A</span>"
            
            # Topic Display Logic
            # Check if this article was in the "AI Verified" batch
            # We don't have a direct flag on the object unless we add it, but we can infer from score/logic
            # Start with basic score display
            ai_status = "🔴" # Default: Heuristic only
            if art in candidates_for_ai: # If it was a candidate (implies it passed pre-filter)
                 # If it's in the final list, it Passed AI (or AI failed closed to True)
                 ai_status = "🤖"
            
            topic_display = f"{ai_status} {s['topic']}"
            
            # Score Styling
            if art.relevance_score > 80:
                score_bg = "#052e16" # Dark Green
                score_text = "#4ade80" # Bright Green
                score_border = "#15803d"
            elif art.relevance_score > 50:
                score_bg = "#422006" # Dark Yellow/Brown
                score_text = "#facc15" # Bright Yellow
                score_border = "#a16207"
            else:
                score_bg = "#450a0a" 
                score_text = "#f87171" 
                score_border = "#b91c1c"
            
            score_badge = f"""
            <span style="display: inline-block; background-color: {score_bg}; color: {score_text}; border: 1px solid {score_border}; padding: 4px 8px; border-radius: 6px; font-weight: bold; min-width: 40px; text-align: center;">
                {art.relevance_score}
            </span>
            """
            
            title_color = "#f1f5f9"
            
            table_html += f"""
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 10px 16px; border-bottom: 1px solid #1e293b;">{score_badge}</td>
                    <td style="padding: 10px 16px; text-align: center; border-bottom: 1px solid #1e293b; font-size: 0.9rem; color: #cbd5e1;">{date_display}</td>
                    <td style="padding: 10px 16px; text-align: center; border-bottom: 1px solid #1e293b; font-size: 0.9rem; color: #cbd5e1;">{topic_display}</td>
                    <td style="padding: 10px 16px; border-bottom: 1px solid #1e293b;">
                        <a href="{art.url}" target="_blank" style="display: block; color: {title_color}; text-decoration: none; font-size: 1rem; font-weight: 500; line-height: 1.4; transition: color 0.2s;">
                           <span style="border-bottom: 1px dotted #94a3b8;">{art.title}</span> <span style="font-size: 0.8em; text-decoration: none;">🔗</span>
                        </a>
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px; font-family: monospace;">{art.url[:80]}{'...' if len(art.url) > 80 else ''}</div>
                    </td>
                </tr>
            """
        
        table_html += "</tbody></table>"

    digest_md = table_html # Return HTML as the 'markdown' response
    analysis_digest = [] # No digest analysis needed since we didn't generate one
    
    # 3. Construct Response
    return DigestResponse(
        digest=digest_md,
        analysis_source=analysis_source,
        analysis_digest=analysis_digest,
        articles=[a.dict() for a in all_articles] # Serialize Pydantic models
    )

# --- Analysis / Playground ---

class AnalysisRequest(BaseModel):
    text: str

@router.post("/outlets/analyze", response_model=List[KeywordData])
async def analyze_digest_text(req: AnalysisRequest, current_user: User = Depends(get_current_user)):
    """
    Direct analysis endpoint (fallback/manual).
    """
    return await analyze_text_with_gemini(req.text, api_key=current_user.gemini_api_key)

