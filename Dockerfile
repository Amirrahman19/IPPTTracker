FROM python:3.11-slim

# mediapipe / opencv need these shared libs even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs

# editable install: config.py resolves PROJECT_ROOT relative to the source
# tree (src/ippt/config.py -> parents[2]), so configs/ must stay alongside it
# rather than being left behind in site-packages by a non-editable install.
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["ippt", "dashboard"]
