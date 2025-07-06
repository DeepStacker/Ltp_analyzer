from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from datetime import datetime
import json
import httpx  # async HTTP client

from logicAlgo import RobustAOCTradingSystem

app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML template rendering
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/analyze", response_class=JSONResponse)
async def analyze_market_data(
    request: Request, authorization: Optional[str] = Header(None)
):
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )

        # Extract Bearer token
        token = authorization.split("Bearer ")[1]

        # Initialize logic system with token
        system = RobustAOCTradingSystem(
            threshold_percentage=0.75, completion_threshold=0.95, token=token
        )

        # Asynchronous data fetch using httpx
        async with httpx.AsyncClient(timeout=5) as client:
            headers = {
                "Accept": "*/*",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            response = await client.get(
                "https://logictrader.in/api/optionchain/NIFTY", headers=headers
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail="Failed to fetch option data"
            )

        option_data = response.json()
        data = {
            "data": {"data": option_data},
        }

        # Optional: Save raw data locally for debugging/logging
        # with open("data.json", "w") as f:
        #     json.dump(data, f)

        # Analyze using your trading logic
        result = system.analyze_market_with_completion_logic(data)

        if result.get("error"):
            return JSONResponse(status_code=400, content={"error": result["error"]})

        return result

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/scenarios")
async def get_scenarios():
    try:
        return FileResponse("scenario-chart.png", media_type="image/png")
    except FileNotFoundError:
        return JSONResponse(
            status_code=404, content={"error": "Scenarios file not found"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
