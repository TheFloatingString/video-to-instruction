# Video To InstructIon demo (vtii)

## To run the latest release as a package

```
set X_OPENAI_API_KEY="<API KEY>"
pip install git+https://github.com/TheFloatingString/video-to-instruction-demo.git
```

## For development 

```bash
set X_OPENAI_API_KEY="<API KEY>"
pip install uv
uv sync
uv run main.py <filepath to video>
```
