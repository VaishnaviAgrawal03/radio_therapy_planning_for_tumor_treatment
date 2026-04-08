"""
FastAPI application for the RadiotherapyPlanningEnv.

Exposes the environment over HTTP endpoints compatible with OpenEnv EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /health: Health check
    - GET /metadata: Environment metadata
    - GET /schema: Action/observation schemas

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI, Request
from server.models import RadiotherapyAction, RadiotherapyObservation
from server.radiotherapy_environment import RadiotherapyEnvironment

app = FastAPI(
    title="RadiotherapyPlanningEnv",
    description="OpenEnv RL environment for cancer radiotherapy treatment planning",
    version="1.0.0",
)

env = RadiotherapyEnvironment()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "RadiotherapyPlanningEnv",
        "description": (
            "An RL environment for cancer radiotherapy treatment planning. "
            "An agent places and optimizes radiation beams to maximize tumor "
            "dose while protecting organs-at-risk."
        ),
        "version": "1.0.0",
        "tasks": ["prostate", "head_neck", "pediatric_brain"],
    }


@app.get("/schema")
def schema():
    return {
        "action": RadiotherapyAction.model_json_schema(),
        "observation": RadiotherapyObservation.model_json_schema(),
        "state": {
            "type": "object",
            "description": "Full environment state including dose grid, beams, and patient data",
        },
    }


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    task = body.get("task", "prostate")
    result = env.reset(task=task)
    return result


@app.post("/step")
def step(action: RadiotherapyAction):
    result = env.step(action.model_dump())
    return result


@app.get("/state")
def state():
    return env.state()


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for uv run or python -m execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
