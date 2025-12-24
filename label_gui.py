#!/usr/bin/env python3
import json
import re
from pathlib import Path
from nicegui import ui

# from ministral_test_base_unsloth import MinistralVLM


# ministralVLM = MinistralVLM()

# Load the training data
json_path = Path(__file__).parent / "training_data" / "main.json"
images_dir = Path(__file__).parent / "training_data" / "images"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Current index
current_index = 0


def parse_answer(answer_text):
    """Parse answer text to extract think and answer parts"""
    think_match = re.search(r"<think>(.*?)</think>", answer_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)

    think_text = think_match.group(1).strip() if think_match else ""
    answer_text = answer_match.group(1).strip() if answer_match else answer_text

    return think_text, answer_text


def update_display():
    """Update the displayed content based on current index"""
    entry = data[current_index]
    image_path = images_dir / entry["image"]
    generated_output = ""

    questions_2 = """分類並統計圖中的商品
        輸出格式為
        -商品名稱
        -顏色
        -數量

        例如：
        -商品名稱：麥香綠茶
        -顏色：綠色
        -數量：2瓶

        -商品名稱：蘋果汁
        -顏色：紅色
        -數量：2瓶"""

    # Update image
    if image_path.exists():

        # generated_output = ministralVLM.generate(image_path, questions_2)
        image.set_source(str(image_path))
    else:
        image.set_source("")

    # Update text fields
    question_input.value = entry["question"]

    # Parse and update think and answer
    think_text, answer_text = parse_answer(entry["answer"])
    think_input.value = think_text
    answer_input.value = answer_text

    # ministral_generate_label.text = generated_output

    # Update counter
    print(current_index)
    counter_label.text = f"Image {current_index + 1} / {len(data)}"

    # Update button states
    prev_button.enabled = current_index > 0
    next_button.enabled = current_index < len(data) - 1


def next_image():
    """Go to next image"""
    global current_index
    save_data()
    if current_index < len(data) - 1:
        current_index += 1
        update_display()


def prev_image():
    """Go to previous image"""
    global current_index
    if current_index > 0:
        current_index -= 1
        update_display()


def save_data():
    """Save current data to JSON file"""
    with open(json_path, "w", encoding="utf-8") as f:
        entry = data[current_index]

        entry["question"] = question_input.value
        entry["answer"] = answer_input.value

        print(entry["answer"])
        print(entry["question"])

        json.dump(data, f, ensure_ascii=False, indent=4)


# Create the UI
ui.page_title("Image Labeling Viewer")

with ui.column().classes("w-full p-4"):
    ui.label("Image Labeling Viewer").classes("text-3xl font-bold mb-4 text-center")

    # Counter
    counter_label = ui.label(f"Image 1 / {len(data)}").classes(
        "text-xl mb-4 text-center"
    )

    # Main content: splitter for image and text
    with ui.splitter(value=50).classes("w-full h-[600px]") as splitter:
        # Left side: Image
        with splitter.before:
            with ui.card().classes("w-full h-full"):
                image = ui.image("").classes("w-full")

        # Right side: Question, Think and Answer
        with splitter.after:
            with ui.column().classes("w-full h-full gap-4 p-4"):
                # Question section
                with ui.card().classes("w-full"):
                    ui.label("Question:").classes("text-lg font-bold mb-2")
                    question_input = (
                        ui.textarea(placeholder="Enter your text...")
                        .props("autogrow")
                        .classes("w-full")
                    )

                # Think section
                with ui.card().classes("w-full"):
                    ui.label("Think:").classes("text-lg font-bold mb-2")
                    think_input = (
                        ui.textarea(placeholder="Enter your text...")
                        .props("autogrow")
                        .classes("w-full")
                    )

                # Answer section
                with ui.card().classes("w-full"):
                    ui.label("Answer:").classes("text-lg font-bold mb-2")
                    answer_input = (
                        ui.textarea(placeholder="Enter your text...")
                        .props("autogrow")
                        .classes("w-full")
                    )

                # Ministral Generate label
                with ui.card().classes("w-full"):
                    ui.label("Ministral Generate:").classes("text-lg font-bold mb-2")
                    ministral_generate_label = ui.label("").classes(
                        "text-base whitespace-pre-wrap"
                    )

    # Navigation buttons
    with ui.row().classes("gap-4 justify-center mt-4"):
        prev_button = ui.button("← Previous", on_click=prev_image).classes("px-6 py-2")
        next_button = ui.button("Next →", on_click=next_image).classes("px-6 py-2")
        save_button = ui.button("Save", on_click=save_data).classes("px-6 py-2")

# Initialize display
update_display()


# Run the app
ui.run(title="Image Labeling Viewer", port=8080, reload=False)
