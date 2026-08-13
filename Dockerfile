FROM python:3.11-slim

WORKDIR /app

# System deps for lxml/playwright-adjacent scraping libs some crewai_tools
# extras pull in transitively.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p outputs .cache

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--demo"]
