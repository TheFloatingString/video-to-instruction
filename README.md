# Video To InstructIon demo (vtii)

## To run the latest release as a package

```bash
set X_OPENAI_API_KEY="<API KEY>"
pip install git+https://github.com/TheFloatingString/video-to-instruction-demo.git
```

And then in Python:
```python
from vtii import video_to_instruction
video_filepath:str = "<video filepath>"
video_mode:str = "<video mode>"
instruction:str =  video_to_instruction(video_filepath, video_mode)
```

## For development 

```bash
set X_OPENAI_API_KEY="<API KEY>"
pip install uv
uv sync
uv run main.py <filepath to video> <video_mode>
```
