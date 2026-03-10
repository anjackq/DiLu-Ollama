import re
from rich import print
import gradio as gr
import yaml
import argparse
from dilu.scenario.envScenarioReplay import EnvScenarioReplay
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.runtime import configure_runtime_env

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

# --- Configuration Setup ---
selected_model = configure_runtime_env(config)
if config['OPENAI_API_TYPE'] == 'ollama':
    print(f"[bold yellow]Visualizer Configured for Local Ollama: {selected_model}[/bold yellow]")
elif config['OPENAI_API_TYPE'] == 'gemini':
    print(f"[bold yellow]Visualizer Configured for Gemini API: {selected_model}[/bold yellow]")

TAMDTemplate = """
# Thoughts and Actions

The following sentences are the **Thoughts and Actions** made by **Driver Agent** at the decision frame {}. It may be incorrect, and if it ultimately leads to a conflict, please modify the wrong part and then click the `Commit Experience` button to submit the changes made. These changes will be used to guide **Driver Agent** to make correct decisions in the future.

{}

{}
"""


def viewFrame(decisionFrame):
    # Ensure input is int for internal logic
    decisionFrame = int(decisionFrame)
    imd = esr.plotSce(decisionFrame)
    framePrompts = esr.getPrompts(decisionFrame)
    if framePrompts.done:
        doneString = "The decision for this frame failed, resulting in subsequent collisions."
    else:
        doneString = "The decision for this frame was successful, and the vehicle did not collide."

    if framePrompts.editTimes:
        editedTimeString = f"Edited times: {framePrompts.editTimes}."
    else:
        editedTimeString = ""

    TAMDStr = TAMDTemplate.format(
        decisionFrame, doneString, editedTimeString,
    )

    # Use edited thoughts if they exist
    final_ta = framePrompts.editedTA if framePrompts.editedTA else framePrompts.thoughtsAndAction

    return (
        imd, framePrompts.description,
        framePrompts.fewshots, TAMDStr,
        final_ta
    )


def nextFramePrompts(decisionFrame):
    nextFrame = int(decisionFrame) + 1
    if nextFrame <= maxFrame:
        imd, descriptionStr, fewshotsStr, TAMDStr, TAStr = viewFrame(nextFrame)
        # Return str(nextFrame) to update the Dropdown
        return str(nextFrame), imd, descriptionStr, fewshotsStr, TAMDStr, TAStr
    else:
        raise gr.Error(f'The range of Decision Frame is {minFrame}~{maxFrame}.')


def lastFramePrompts(decisionFrame):
    lastFrame = int(decisionFrame) - 1
    if lastFrame >= 0:
        imd, descriptionStr, fewshotsStr, TAMDStr, TAStr = viewFrame(lastFrame)
        return str(lastFrame), imd, descriptionStr, fewshotsStr, TAMDStr, TAStr
    else:
        raise gr.Error(f'The range of Decision Frame is {minFrame}~{maxFrame}.')


def commitExperience(decisionFrame, expertExperience):
    try:
        decisionFrame = int(decisionFrame)
        framePrompts = esr.getPrompts(decisionFrame)

        # Regex to extract scenario
        pattern = r"#### Driving scenario description:(.*?)####"
        match = re.search(pattern, framePrompts.description, re.DOTALL)
        if match:
            sce_descrip = match.group(1).strip()
        else:
            raise gr.Error("Cannot find Driving scenario description in prompt.")

        # Regex to extract action
        pattern = r"Response to user:#### (\d+)"
        match = re.search(pattern, expertExperience)
        if match:
            action = int(match.group(1))
            print("action: ", action)
        else:
            raise gr.Error("Please make sure the last line contains 'Response to user:#### <Action_ID>'.")

        vector_memory.addMemory(
            sce_descrip, framePrompts.description, expertExperience, action)

        esr.editTA(decisionFrame, expertExperience)
        gr.Info('The Thoughts and Actions has been edited and committed.')

        # Refresh view
        _, _, _, TAMDStr, TAStr = viewFrame(decisionFrame)
        return TAMDStr, TAStr
    except Exception as e:
        gr.Error(f'Error committing experience: {str(e)}')
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizer for DiLu Results")
    parser.add_argument("-r", "--result_db_path", type=str, help="Path to result db", required=True)
    parser.add_argument("-m", "--mem_path", type=str, help="Path to memory db", required=True)
    args = parser.parse_args()

    esr = EnvScenarioReplay(args.result_db_path)
    minFrame, maxFrame = esr.getMinMaxFrame()
    vector_memory = DrivingMemory(db_path=args.mem_path)

    with gr.Blocks(theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg)) as demo:
        with gr.Row(visible=True, variant='panel'):
            # FIX 1: Convert range to list of strings to match Dropdown value type
            frame_values = [str(f) for f in range(minFrame, maxFrame + 1)]

            decisionFrame = gr.Dropdown(
                choices=frame_values,
                value=str(minFrame),
                label="Decision Frame",
                interactive=True
            )
            viewerBtn = gr.Button(scale=1, value='View Scenario')
            lastFrameBtn = gr.Button(scale=1, value="Last Frame")
            nextFrameBtn = gr.Button(scale=1, value="Next Frame")

        with gr.Row(visible=True, variant='panel'):
            currentImage = gr.Image(interactive=False, scale=1)
            with gr.Column():
                DesMD = gr.Markdown("# Driving scenario description")
                descriptionText = gr.TextArea(scale=1, interactive=False, lines=28, label="")

        with gr.Row(visible=True, variant='panel'):
            with gr.Column():
                FSMD = gr.Markdown("# Few-shot")
                fewShotsText = gr.TextArea(scale=1, interactive=False, label="", lines=35)
            with gr.Column():
                TAMD = gr.Markdown("# Thoughts and Actions")
                TAText = gr.TextArea(scale=1, interactive=True, lines=28, label="")

        commitBtn = gr.Button(value='Commit Experience')

        # Event Listeners
        viewerBtn.click(
            viewFrame,
            inputs=[decisionFrame],
            outputs=[currentImage, descriptionText, fewShotsText, TAMD, TAText],
        )
        lastFrameBtn.click(
            lastFramePrompts,
            inputs=[decisionFrame],
            outputs=[decisionFrame, currentImage, descriptionText, fewShotsText, TAMD, TAText],
        )
        nextFrameBtn.click(
            nextFramePrompts,
            inputs=[decisionFrame],
            outputs=[decisionFrame, currentImage, descriptionText, fewShotsText, TAMD, TAText],
        )
        commitBtn.click(
            commitExperience,
            inputs=[decisionFrame, TAText],
            outputs=[TAMD, TAText],
        )

    # FIX 2: Removed deprecated concurrency_count from queue()
    demo.queue()
    demo.launch()
