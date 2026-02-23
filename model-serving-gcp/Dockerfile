# Zainstaluj pythona
FROM python:3.14

# Wybierz katalog gdzie skopiujesz pliki
WORKDIR /app

# Skopiuj plik z zależnościami
COPY requirements.txt requirements.txt

# Zainstaluj zależności za pomocą pip
RUN pip install --no-cache -r requirements.txt

# Skopiuj kod aplikacji
COPY app.py app.py
COPY model.joblib model.joblib
COPY index.html index.html

# Uruchom aplikacje
CMD ["fastapi", "run", "app.py"]

# BUDOWANIE OBRAZU
# docker build -t model-server:v0.1 .

# URUCHAMIANIE KONTENERA Z OBRAZU
# docker run -p 8000:8000 model-server:v0.1
