from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.pdf import router as pdf_router
from app.routes.qa import router as qa_router



def create_app() -> FastAPI:
    app = FastAPI(title="Local RAG PDF QA Backend", version="0.1.0")
    app.include_router(health_router)
    app.include_router(pdf_router)

    app.include_router(qa_router)
    return app


app = create_app()

