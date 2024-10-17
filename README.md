# MVPCD - Automatisierte YOLO-Annotierung Pipeline

## Projektübersicht

MVPCD ist ein Projekt zur automatischen Erstellung von Annotierungen für YOLO-Modelle. Die Pipeline umfasst das Erfassen von Bildern mit einer Zed-Kamera, Vorverarbeitung der Bilder mittels Chroma Keying und Tiefeninformationen, sowie die automatische Erstellung von Bounding Boxes für verschiedene Objekte.

## Projektstruktur

MVPCD/ 
├── README.md 
├── requirements.txt 
| ├── data/ 
│ ├── images.py 
│ └── labels/ 
├── scripts/ 
│ ├── capture.py 
│ ├── preprocess.py │ 
| ├── annotate.py 
│ └── main.py 
├── config/ 
│ └── config.yaml 
└── utils/ 
│ ├── chroma_key.py 
│ ├── depth_processing.py 
│ └── bbox_utils.py


## Installation

1. **Clone das Repository:**

   ```bash
   cd ~/Desktop
   git clone <repository_url> MVPCD
   cd MVPCD

2. **Erstelle und aktiviere eine virtuelle Umgebung:**
   python3 -m venv venv
   source venv/bin/activate

3. **Installiere die Abhängigkeiten:**
   pip install -r requirements.txt

## Nutzung
1. **Konfiguriere die Einstellungen**
   Bearbeite die Datei config/config.yaml nach deinen Bedürfnissen.

2. **Führe das Hauptskript aus**
   python scripts/main.py

## Komponenten

- *capture.py:* Erfassen von Bildern und Tiefendaten von der Zed-Kamera.
- *preprocess.py:* Anwenden von Chroma Keying und ROI-Definition.
- *annotate.py:* Automatisches Erstellen von Bounding Boxes basierend auf Tiefendaten.
- *main.py:* Orchestrierung der gesamten Pipeline.
- *utils/:* Enthält Hilfsfunktionen für verschiedene Schritte der Pipeline.

## Anforderungen
   Siehe requirements.txt für eine vollständige Liste der benötigten Python-Pakete.

## Lizenz
    Dieses Projekt ist unter der MIT-Lizenz lizenziert.