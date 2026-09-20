"""Local API: inspection input is confined to the configured raw-data directory."""
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from vision_sorter.config import Config
from vision_sorter.classification.rules import RuleConfig
from vision_sorter.inspection.inspection_service import InspectionService


class InspectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    save_intermediate: bool = False
    invert: bool = False


def create_app(config: Config | None = None, rules: RuleConfig | None = None) -> FastAPI:
    settings = config or Config()
    service = InspectionService(settings, rules)
    app = FastAPI(title="Vision Sorter", version="0.1.0", description="Classical CV and mock sorting; local use only")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "detector": "classical", "actuator": "mock"}

    @app.get("/summary")
    def summary() -> dict:
        return service.repository.summary()

    @app.get("/inspections")
    def inspections(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> list[dict]:
        return service.repository.list(limit, offset)

    @app.get("/inspections/{identifier}")
    def inspection(identifier: str) -> dict:
        result = service.repository.get(identifier)
        if result is None:
            raise HTTPException(404, "Inspection not found")
        return result

    @app.post("/inspections", status_code=201)
    def inspect(request: InspectionRequest) -> dict:
        raw = (settings.data_dir / "raw").resolve()
        candidate = (raw / request.path).resolve()
        if not candidate.is_relative_to(raw):
            raise HTTPException(400, "Input path must be inside data/raw")
        if not candidate.is_file():
            raise HTTPException(404, "Input image not found")
        if candidate.suffix.lower() not in settings.image_extensions:
            raise HTTPException(400, "Unsupported image extension")
        worker = InspectionService(settings, rules, invert=request.invert)
        result = worker.inspect_path(candidate, request.save_intermediate)
        return worker.repository.get(result.id)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return """<!doctype html><html lang="en"><meta charset="utf-8"><title>Vision Sorter</title>
        <style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px;background:#111827;color:#eee}
        table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #475569;padding:12px}
        a{color:#67e8f9}button,input{font:inherit;padding:10px;margin:4px}#summary{padding:20px;background:#1e293b}</style>
        <h1>Vision Sorter</h1><p>Classical computer vision · Mock sorting</p>
        <p>Enter an image path relative to data/raw.</p><form id="form"><input id="path" required placeholder="sample.png">
        <button>Inspect image</button></form><p id="message" role="status"></p>
        <p id="summary">Loading…</p><button id="refresh">Refresh</button> <a href="/docs">API documentation</a>
        <table><thead><tr><th>Status</th><th>Reason</th><th>Lane</th><th>Source</th></tr></thead><tbody id="rows"></tbody></table>
        <script>
        async function request(url, options){const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw Error(d.detail||r.statusText);return d;}
        async function refresh(){try{const [s,rows]=await Promise.all([request('/summary'),request('/inspections')]);
          document.getElementById('summary').textContent=Object.entries(s).map(([k,v])=>k+': '+v).join(' | ');
          const body=document.getElementById('rows');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');
          for(const key of ['status','reason','sort_lane','source']){const td=document.createElement('td');td.textContent=row[key];tr.append(td);}body.append(tr);}}
          catch(e){document.getElementById('message').textContent=e.message;}}
        document.getElementById('refresh').onclick=refresh;
        document.getElementById('form').onsubmit=async e=>{e.preventDefault();try{const d=await request('/inspections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:document.getElementById('path').value})});document.getElementById('message').textContent=d.status+': '+d.reason;await refresh();}catch(err){document.getElementById('message').textContent=err.message;}};
        refresh();</script></html>"""

    return app
