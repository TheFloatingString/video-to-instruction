import cv2
import numpy as np
import os
import argparse
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import openai
from threading import Thread
import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv("X_OPENAI_API_KEY")

VERBOSE = False

DOWNSIZE_FRAMES = True

CONFIG_PROMPTS = {
    "single_task": {
        "frame_prompt": "what is in the following video frame? Specifically mention if there are hands present, and what the hands are doing. ",
        "summary_prompt": "The following are textual descriptions of video frames, separated by a '---'. What is one sentence that summarizes an action that a robot needs to take to perform the action, knowing that the hand represents what the robot arm should be doing? Only respond with a single sentence that describes the robot action, in a sentence that follows the following structure: <object> <verb> <descriptor>. \n",
    },
    "multiple_tasks": {
        "frame_prompt": "what is in the following video frame? Specifically mention if there are hands present, and what the hands are doing. Also describe each item in the frame, and its colour. Give each object a unique and intuitive descriptor. Mention what unique objects the hands are acting on, if present.",
        "summary_prompt": "The following are textual descriptions of video frames, separated by a '---'.  Determine how many actions are represented in the following descriptions. For each action, use a short and concise sentence with the following structure: <object> <verb> <descriptor>. Return with a numbered list, but nothing else. \n",
    },
}


main_list = []


def numpy_to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_img}"


def get_description_for_frame(frame: np.ndarray, prompt_mode: str) -> str:
    image_data_url = numpy_to_base64(frame)
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": CONFIG_PROMPTS[prompt_mode]["frame_prompt"],
            },
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": image_data_url}],
            },
        ],
    )
    if VERBOSE:
        print(resp.output[0].content[0].text)
    return resp.output[0].content[0].text


def get_summary_from_frame_descriptions(
    list_of_descriptions: list[str], prompt_mode: str
) -> str:
    prompt_str = CONFIG_PROMPTS[prompt_mode]["summary_prompt"]
    for descr in list_of_descriptions:
        prompt_str += f"{descr}\n---"
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "user", "content": prompt_str},
        ],
    )
    return resp.output[0].content[0].text


def populate_image_content(frames):
    content = []

    for idx, frame in enumerate(frames):
        content.append({"type": "input_text", "text": f"image {idx}"})
        content.append({"type": "input_image", "image_url": numpy_to_base64(frame)})

    return content


def get_summary_from_windows(video_filepath: str) -> str:
    # list_of_action_summaries = []
    cap = cv2.VideoCapture(video_filepath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if VERBOSE:
        print("video loaded.")
    window_length = 120
    n_windows = int(
        2 * len(frames) / window_length
    )  # TODO: there might be clipping here
    window_step_size = int(window_length / 2)  # TODO: there might be clipping here
    global main_list
    main_list = [None] * n_windows
    start = 0
    end = window_step_size

    threads = []
    main_list_idx = 0

    for i in tqdm.trange(n_windows):
        frames_window = frames[start:end]
        threads.append(
            Thread(
                target=thread_get_summary_from_sliding_window_frame,
                args=(frames_window, main_list_idx),
            )
        )
        main_list_idx += 1

    N_WORKERS = 50
    if n_windows < N_WORKERS:
        N_WORKERS = n_windows

    for i in tqdm.trange(int(n_windows / N_WORKERS)):
        if VERBOSE:
            print(i)
        sublist_threads = threads[:N_WORKERS]
        for t in sublist_threads:
            t.start()
        for t in sublist_threads:
            t.join()
        for i in range(N_WORKERS):
            threads.pop(0)
            if VERBOSE:
                print(threads)

    for i in tqdm.trange(len(threads)):
        threads[i].start()
    for i in tqdm.trange(len(threads)):
        threads[i].join()

        # action_summary = get_summary_from_sliding_window_frame(frames_window)
        # print(action_summary)
        # start += window_step_size
        # end += window_step_size
        # list_of_action_summaries.append(action_summary)
    return main_list


def downsize_frame(frame):
    return cv2.resize(frame, (224, 224))


def thread_get_summary_from_sliding_window_frame(
    list_of_frames: list, main_list_idx: int
) -> str:
    global main_list
    frames = []

    for i in range(len(list_of_frames)):
        if i % 15 == 0:
            frames.append(list_of_frames[i])

    if DOWNSIZE_FRAMES:
        frames = [downsize_frame(frame) for frame in frames]

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    if VERBOSE:
        print(len(frames))
    resp = client.responses.create(
        model="o4-mini",
        input=[
            {
                "role": "user",
                "content": """Given the following frames:

- what is the image number where the [action] begins?
- what is the image number where the [action] ends?

[action] = picking up and putting down the can""",
            },
            {
                "role": "user",
                "content": populate_image_content(frames),
            },
        ],
    )

    if VERBOSE:
        print(resp.output[-1].content[0].text)
    main_list[main_list_idx] = resp.output[-1].content[0].text
    # return resp.output[-1].content[0].text


def get_summary_from_sliding_window_frame(list_of_frames: list) -> str:
    frames = []

    for i in range(len(list_of_frames)):
        if i % 15 == 0:
            frames.append(list_of_frames[i])

    if DOWNSIZE_FRAMES:
        frames = [downsize_frame(frame) for frame in frames]

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    if VERBOSE:
        print(len(frames))
    resp = client.responses.create(
        model="o4-mini",
        input=[
            {
                "role": "user",
                "content": """Given the following frames:

- what is the image number where the [action] begins?
- what is the image number where the [action] ends?

[action] = picking up and putting down the can""",
            },
            {
                "role": "user",
                "content": populate_image_content(frames),
            },
        ],
    )

    if VERBOSE:
        print(resp.output[-1].content[0].text)
    return resp.output[-1].content[0].text


def get_description_from_window(video_frames: list) -> str:
    pass


def get_summary_from_video(video_filepath: str) -> str:
    cap = cv2.VideoCapture(video_filepath)
    counter = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        counter += 1
        if counter % 15 == 0:
            frames.append(frame)

    for i in tqdm.trange(len(frames)):
        cv2.imwrite(f"tmp-{i}.png", frames[i])

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    if VERBOSE:
        print(len(frames))
    resp = client.responses.create(
        model="o4-mini",
        input=[
            {
                "role": "user",
                "content": """Given the following frames:

- what is the image number where the [action] begins?
- what is the image number where the [action] ends?

[action] = picking up and putting down the can""",
            },
            {
                "role": "user",
                "content": populate_image_content(frames),
            },
        ],
    )

    if VERBOSE:
        print(resp.output[-1].content[0].text)
    return resp.output[-1].content[0].text


def video_to_instruction(filepath: str, prompt_mode: str) -> str:
    if prompt_mode == "video_to_instructions":
        instruction: str = get_summary_from_video(filepath)
        return instruction
    if prompt_mode == "sliding_window":
        instruction: str = get_summary_from_windows(filepath)
        return instruction
    else:
        cap = cv2.VideoCapture(filepath)
        counter = 0
        list_of_frame_descriptions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1
            if counter % 20 == 0:
                frame_description = get_description_for_frame(frame, prompt_mode)
                list_of_frame_descriptions.append(frame_description)
                if VERBOSE:
                    print(".", end="")
        if VERBOSE:
            print()
        instruction = get_summary_from_frame_descriptions(
            list_of_frame_descriptions, prompt_mode
        )
        if VERBOSE:
            print(instruction)
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("prompt_mode")
    args = parser.parse_args()
    video_to_instruction(args.filepath, args.prompt_mode)
