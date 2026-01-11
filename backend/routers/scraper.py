
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import List, Optional
from pydantic import BaseModel
import json
from database import AsyncSessionLocal
from models import ScraperRule
import scraper_engine

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
import httpx

router = APIRouter()

# --- Schemas ---

class ScraperRuleCreate(BaseModel):
    domain: str
    date_selectors: Optional[List[str]] = None
    date_regex: Optional[List[str]] = None
    use_json_ld: bool = True
    use_data_layer: bool = False
    data_layer_var: Optional[str] = None

class ScraperRuleRead(BaseModel):
    id: int
    domain: str
    config: dict # Normalized config
    
    class Config:
        from_attributes = True

class TestExtractionRequest(BaseModel):
    url: str
    rule_config: Optional[ScraperRuleCreate] = None # Optional override

# --- Endpoints ---

@router.get("/scraper/rules", response_model=List[ScraperRuleRead])
async def list_rules(db: Session = Depends(get_db)):
    result = await db.execute(select(ScraperRule))
    rules = result.scalars().all()
    results = []
    for r in rules:
        results.append(ScraperRuleRead(
            id=r.id,
            domain=r.domain,
            config=json.loads(r.config_json) if r.config_json else {}
        ))
    return results

@router.post("/scraper/rules", response_model=ScraperRuleRead)
async def create_or_update_rule(rule: ScraperRuleCreate, db: Session = Depends(get_db)):
    # Check existing
    result = await db.execute(select(ScraperRule).where(ScraperRule.domain == rule.domain))
    existing = result.scalar_one_or_none()
    
    config_dict = rule.dict(exclude={"domain"})
    
    if existing:
        existing.config_json = json.dumps(config_dict)
        await db.commit()
        await db.refresh(existing)
        return ScraperRuleRead(id=existing.id, domain=existing.domain, config=config_dict)
    else:
        new_rule = ScraperRule(
            domain=rule.domain,
            config_json=json.dumps(config_dict)
        )
        db.add(new_rule)
        await db.commit()
        await db.refresh(new_rule)
        return ScraperRuleRead(id=new_rule.id, domain=new_rule.domain, config=config_dict)

@router.post("/scraper/test")
async def test_extraction(req: TestExtractionRequest, db: Session = Depends(get_db)):
    """
    Test extraction live. 
    If rule_config is provided, it uses that ephemeral config.
    If not, it looks up DB/Registry for the URL's domain.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,uk;q=0.8",
        "Referer": "https://www.google.com/",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
    }
    async with httpx.AsyncClient(follow_redirects=True, verify=False, http2=True) as client:
        try:
            resp = await client.get(req.url, headers=headers, timeout=15)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to fetch URL: Status {resp.status_code}")
            
            html = resp.text
            
            # Create Ephemeral Rule if provided
            custom_rule = None
            if req.rule_config:
                from scraper_engine import ScraperRule as EngineRule
                custom_rule = EngineRule(
                    domain="test",
                    date_selectors=req.rule_config.date_selectors,
                    date_regex=req.rule_config.date_regex,
                    use_json_ld=req.rule_config.use_json_ld,
                    use_data_layer=req.rule_config.use_data_layer,
                    data_layer_var=req.rule_config.data_layer_var
                )
            else:
                 # Fetch from DB if no config provided
                 from urllib.parse import urlparse
                 domain = urlparse(req.url).netloc.replace("www.", "").lower()
                 
                 result = await db.execute(select(ScraperRule).where(ScraperRule.domain == domain))
                 db_rule = result.scalar_one_or_none()
                 
                 if db_rule:
                      config = json.loads(db_rule.config_json)
                      from scraper_engine import ScraperRule as EngineRule
                      custom_rule = EngineRule(
                          domain=domain,
                          date_selectors=config.get('date_selectors'),
                          date_regex=config.get('date_regex'),
                          use_json_ld=config.get('use_json_ld', True),
                          use_data_layer=config.get('use_data_layer', False),
                          data_layer_var=config.get('data_layer_var')
                      )
            
            # Extract
            extracted_date = scraper_engine.extract_date_from_html(html, req.url, custom_rule_override=custom_rule)
            
            return {
                "status": "success",
                "extracted_date": extracted_date,
                "used_rule": req.rule_config or "System Default"
            }
            
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
