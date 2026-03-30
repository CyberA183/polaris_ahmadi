## POLARIS
- - -
A multi-agent AI framework that utilizes LLMs to integrate experimental analysis, literature review,
and simulation agents in order to automate and accelerate materials science research

### 
- - -
**How to Run App on Machine:** <br>
1. Install Requirements
```commandline 
pip install -r requirements.txt
```
2. Run App
```commandline
streamlit run streamlit_app.py
```

### macOS updater release contract
- - -
The packaged macOS app can check GitHub Releases for updates and stage a new bundle
for installation on next launch.

Expected release assets:

1. `Polaris.app.zip`
2. `version.json`

Recommended tag format:

```text
v1.0.1
```

Example `version.json`:

```json
{
  "version": "1.0.1",
  "notes": "Bug fixes and updater support",
  "macos": {
    "url": "https://github.com/CyberA183/polaris_ahmadi/releases/download/v1.0.1/Polaris.app.zip"
  }
}
```

Recommended hosted manifest URL:

```text
https://github.com/CyberA183/polaris_ahmadi/releases/latest/download/version.json
```