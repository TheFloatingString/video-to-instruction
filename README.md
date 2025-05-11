# Video To InstructIon demo (vtii)

## To run the latest release as a package

```bash
set X_OPENAI_API_KEY="<API KEY>"
pip install git+https://github.com/TheFloatingString/video-to-instruction.git
```

And then in Python:
```python
from vtii import video_to_instruction
video_filepath:str = "<video filepath>"
video_mode:str = "<video mode>"
instruction:str =  video_to_instruction(video_filepath, video_mode)
```

List of video modes:

- `single_task`: only returns a single task, text as context
- `multiple_tasks`: returns multiple tasks, text as context
- `video_to_instructions`: feeds an entire video to OpenAI
- `sliding_window`: sliding window approach

## For development

```bash
set X_OPENAI_API_KEY="<API KEY>"
set SERVER_URI="<URI>"
pip install uv
uv sync
uv run main.py <filepath to video> <video_mode>
```
