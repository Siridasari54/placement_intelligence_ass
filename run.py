import os
import uvicorn

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    host = os.getenv(
        "HOST",
        "127.0.0.1"
    )

    port = int(
        os.getenv(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )