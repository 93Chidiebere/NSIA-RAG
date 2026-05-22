from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from rag_utils import generate_response


# =========================
# INITIALIZE FASTAPI
# =========================

app = FastAPI()


# =========================
# HTML TEMPLATE CONFIG
# =========================

templates = Jinja2Templates(
    directory="templates"
)


# =========================
# HOME PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "response": None
        }
    )


# =========================
# QUESTION ENDPOINT
# =========================

@app.post("/ask", response_class=HTMLResponse)
def ask_question(
    request: Request,
    query: str = Form(...)
):

    # Generate RAG response
    response = generate_response(query)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "response": response
        }
    )